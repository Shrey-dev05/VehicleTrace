"""
Vehicle and Plate Multi-Object Tracking Engine
Associates vehicle and license plate bounding boxes temporally across video frames.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict


def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


def is_box_contained(inner_box: List[int], outer_box: List[int], margin: int = 15) -> bool:
    """Check if inner_box (plate) is geometrically within or intersecting outer_box (vehicle)."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    # Center of plate
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0

    return (ox1 - margin <= cx <= ox2 + margin) and (oy1 - margin <= cy <= oy2 + margin)


class SpatialVehicleTracker:
    """
    Robust Spatial Tracker with IoU matching for vehicles across consecutive frames.
    Assigns local track IDs valid within the video session.
    """
    def __init__(self, iou_threshold: float = 0.35, max_lost_frames: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.tracks = {}  # track_id -> {"bbox": [...], "lost": int, "history": [...]}

    def update(self, detected_bboxes: List[List[int]]) -> List[Tuple[str, List[int]]]:
        """
        detected_bboxes: list of [x1, y1, x2, y2]
        Returns: list of (track_id_str, bbox)
        """
        matched_results = []
        unmatched_dets = list(range(len(detected_bboxes)))
        active_track_ids = list(self.tracks.keys())

        if active_track_ids and detected_bboxes:
            # Build cost matrix
            iou_matrix = np.zeros((len(active_track_ids), len(detected_bboxes)))
            for i, tid in enumerate(active_track_ids):
                for j, bbox in enumerate(detected_bboxes):
                    iou_matrix[i, j] = calculate_iou(self.tracks[tid]["bbox"], bbox)

            matched_tracks = set()
            matched_dets = set()

            # Greedy match by highest IoU
            while True:
                max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = active_track_ids[i]
                bbox = detected_bboxes[j]

                self.tracks[tid]["bbox"] = bbox
                self.tracks[tid]["lost"] = 0
                self.tracks[tid]["history"].append(bbox)
                matched_results.append((tid, bbox))

                matched_tracks.add(tid)
                matched_dets.add(j)
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

            unmatched_dets = [j for j in range(len(detected_bboxes)) if j not in matched_dets]

            # Mark lost tracks
            for tid in active_track_ids:
                if tid not in matched_tracks:
                    self.tracks[tid]["lost"] += 1

        # Delete expired tracks
        to_delete = [tid for tid, data in self.tracks.items() if data["lost"] > self.max_lost_frames]
        for tid in to_delete:
            del self.tracks[tid]

        # Initialize new tracks for unmatched detections
        for j in unmatched_dets:
            bbox = detected_bboxes[j]
            tid = f"TRK-{self.next_track_id:03d}"
            self.next_track_id += 1
            self.tracks[tid] = {"bbox": bbox, "lost": 0, "history": [bbox]}
            matched_results.append((tid, bbox))

        return matched_results


def associate_plates_to_vehicles(
    tracked_vehicles: List[Tuple[str, List[int]]],
    detected_plates: List[Dict[str, Any]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Associate detected plates with the enclosing or nearest vehicle track.
    Returns: {track_id: plate_dict or None}
    """
    associations = {tid: None for tid, _ in tracked_vehicles}
    assigned_plates = set()

    for tid, v_box in tracked_vehicles:
        best_plate = None
        best_overlap = 0.0

        for idx, p in enumerate(detected_plates):
            if idx in assigned_plates:
                continue
            p_box = p["bbox"]
            if is_box_contained(p_box, v_box):
                overlap = calculate_iou(p_box, v_box)
                if best_plate is None or overlap > best_overlap:
                    best_plate = p
                    best_overlap = overlap
                    best_idx = idx

        if best_plate is not None:
            associations[tid] = best_plate
            assigned_plates.add(best_idx)

    return associations
