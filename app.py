import os
import cv2
import json
import time
import numpy as np
import faiss
import csv
from insightface.app import FaceAnalysis
import streamlit as st
from io import StringIO
from tempfile import TemporaryDirectory
import onnxruntime as ort
import pandas as pd
 
# Constants / Configuration
MATCH_THRESHOLD = 1.4
 
# Streamlit Page Config
st.set_page_config(page_title="Face Attendance System", layout="wide")
st.title("Face Attendance System")
 
# Sidebar Configuration
st.sidebar.header("Configuration")
run_button = st.sidebar.button("Run Attendance Detection")
 
# File Uploads
st.sidebar.markdown("### Upload Reference Faces")
ref_files = st.sidebar.file_uploader(
    "Upload one or more reference face images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
 
st.sidebar.markdown("### Upload Meeting Photos")
meeting_files = st.sidebar.file_uploader(
    "Upload one or more meeting images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
 
# Helper: Convert uploaded files to OpenCV images
def file_to_cv2(file):
    file.seek(0)
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
 
# ------------------------ MAIN EXECUTION ------------------------
if run_button:
    if not ref_files or not meeting_files:
        st.error("Please upload both reference and meeting images before running.")
    else:
        start_time = time.time()
        st.info("⏳ Loading model... please wait")
 
        # Dynamic GPU/CPU Selection
        if ort.get_device() == "GPU":
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
 
        # Setup InsightFace
        app = FaceAnalysis(providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        st.success("✅ InsightFace model loaded successfully")
 
        with TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)
 
            # Step 1: Build Reference Embeddings
            st.write("### Building reference embeddings...")
            reference_embeddings = []
            reference_names = []
 
            progress_ref = st.progress(0)
            for i, file in enumerate(ref_files):
                name = os.path.splitext(file.name)[0]
                img = file_to_cv2(file)
                if img is None:
                    st.warning(f"⚠️ Could not read {file.name}, skipping.")
                    continue
 
                faces = app.get(img)
                if not faces:
                    st.warning(f"⚠️ No face found in {file.name}, skipping.")
                    continue
 
                embedding = faces[0].embedding
                norm_embedding = embedding / np.linalg.norm(embedding)
                reference_embeddings.append(norm_embedding)
                reference_names.append(name)
 
                progress_ref.progress((i + 1) / len(ref_files))
 
            if not reference_embeddings:
                st.error("❌ No valid reference faces detected.")
                st.stop()
 
            reference_embeddings = np.array(reference_embeddings).astype("float32")
            embedding_dim = reference_embeddings.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(reference_embeddings)
            st.success(f"✅ Loaded {len(reference_names)} reference embeddings.")
 
            # Step 2: Process Meeting Photos
            st.write("### Processing meeting photos...")
            results_data = {}
            all_attendance_records = []
            unidentified_embeddings = []
            progress_meet = st.progress(0)
 
            meeting_image_names = [file.name for file in meeting_files]
 
            for i, file in enumerate(meeting_files):
                img = file_to_cv2(file)
                if img is None:
                    st.warning(f"⚠️ Failed to read {file.name}, skipping.")
                    continue
 
                faces = app.get(img)
                detected_faces_info = []
 
                if faces:
                    for face in faces:
                        embedding = face.embedding
                        norm_embedding = (
                            embedding / np.linalg.norm(embedding)
                        ).astype("float32").reshape(1, -1)
 
                        D, I = index.search(norm_embedding, 1)
                        distance = D[0][0]
 
                        if distance < MATCH_THRESHOLD:
                            name = reference_names[I[0][0]]
                            confidence = max(0, 1.0 - (distance / MATCH_THRESHOLD))
                        else:
                            name = "Unknown"
                            confidence = 0.0
                            unidentified_embeddings.append(norm_embedding)
 
                        if name != "Unknown":
                            all_attendance_records.append((file.name, name))
 
                        detected_faces_info.append({
                            "name": name,
                            "confidence_score": float(confidence),
                            "raw_distance": float(distance)
                        })
 
                results_data[file.name] = detected_faces_info
                progress_meet.progress((i + 1) / len(meeting_files))
 
            # Step 3: JSON Download
            st.download_button(
                "💾 Download results.json",
                data=json.dumps(results_data, indent=4),
                file_name="results.json",
                mime="application/json"
            )
 
            # ---------------- ATTENDANCE SUMMARY (NO IMAGES) ----------------
            st.write("### 📊 Attendance Summary Matrix")
 
            # Build attendance matrix
            attendance_matrix = {
                name: {photo_name: "" for photo_name in meeting_image_names}
                for name in reference_names
            }
 
            for photo_name, detections in results_data.items():
                for det in detections:
                    name = det["name"]
                    if name in attendance_matrix:
                        attendance_matrix[name][photo_name] = "✅"
 
            # Build CSV rows and display rows (no images)
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
 
            header = ["Name", "Final_Status"] + meeting_image_names
            writer.writerow(header)
 
            display_rows = []
            for name in reference_names:
                # Determine presence
                present_in_photos = any(v == "✅" for v in attendance_matrix[name].values())
                final_status = "Present" if present_in_photos else "Absent"
 
                # CSV row: text only
                row = [name, final_status] + [attendance_matrix[name][p] for p in meeting_image_names]
                writer.writerow(row)
 
                # Display row: text only
                display_rows.append({
                    "Name": name,
                    "Final_Status": final_status,
                    **attendance_matrix[name]
                })
 
            attendance_csv_data = csv_buffer.getvalue().encode("utf-8")
 
            st.download_button(
                "💾 Download Attendance Summary (CSV)",
                data=attendance_csv_data,
                file_name="attendance_summary.csv",
                mime="text/csv"
            )
 
            # Display the table (no images)
            df_display = pd.DataFrame(display_rows)
            st.write("### 🧾 Final Attendance Table")
            st.dataframe(df_display)
 
            # ----------------------------------------------------------
 
            # Step 4: Unique unidentified count (unchanged)
            unique_unidentified_count = 0
            if unidentified_embeddings:
                u_embeddings = np.vstack(unidentified_embeddings)
                u_index = faiss.IndexFlatL2(embedding_dim)
                u_index.add(u_embeddings)
 
                processed_indices = set()
                IDENTITY_THRESHOLD = 1.2
 
                for i in range(len(u_embeddings)):
                    if i in processed_indices:
                        continue
                    unique_unidentified_count += 1
                    _, _, neighbor_indices = u_index.range_search(u_embeddings[i:i+1], IDENTITY_THRESHOLD)
                    for neighbor_idx in neighbor_indices:
                        processed_indices.add(neighbor_idx)
 
            total_time = time.time() - start_time
            total_employees = len(reference_names)
 
            present = sorted(set([record[1] for record in all_attendance_records]))
            absent = sorted(set(reference_names) - set(present))
 
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⏱️ Time Taken", f"{total_time:.2f}s")
            col2.metric("👥 Total Employees", total_employees)
            col3.metric("✅ Present", len(present))
            col4.metric("❌ Absent", len(absent))
 
            st.write("### Present Employees")
            st.dataframe(present)
 
            st.write("### Absent Employees")
            st.dataframe(absent)
 
            # Step 5: JSON Results Display
            st.write("---")
            st.write("### JSON Results")
            st.json(results_data)
 
 