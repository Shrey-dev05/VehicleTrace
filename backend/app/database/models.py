"""
SQLAlchemy Models for Vehicle Trace System
"""
import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class CaseRecord(Base):
    __tablename__ = "cases"

    id = Column(String(64), primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    source_type = Column(String(16), nullable=False)  # "image" or "video"
    source_path = Column(String(512), nullable=False)
    status = Column(String(32), default="pending", index=True)  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    fps = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    duration_sec = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    observations = relationship("VehicleObservation", back_populates="case", cascade="all, delete-orphan")
    fusions = relationship("VehicleFusion", back_populates="case", cascade="all, delete-orphan")
    corrections = relationship("InvestigatorCorrection", back_populates="case", cascade="all, delete-orphan")


class VehicleObservation(Base):
    __tablename__ = "vehicle_attribute_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(String(64), nullable=False, index=True)
    frame_index = Column(Integer, default=0)
    frame_timestamp_ms = Column(Float, default=0.0)
    
    # Bounding boxes stored as json string [x1, y1, x2, y2]
    vehicle_bbox = Column(Text, nullable=True)
    vehicle_crop_path = Column(String(512), nullable=True)
    
    plate_bbox = Column(Text, nullable=True)
    plate_crop_path = Column(String(512), nullable=True)
    
    # Attributes
    vehicle_type_value = Column(String(64), default="unknown")
    vehicle_type_score = Column(Float, default=0.0)
    
    colour_value = Column(String(64), default="unknown")
    colour_score = Column(Float, default=0.0)
    colour_alternatives = Column(Text, nullable=True)  # JSON
    
    make_value = Column(String(64), default="Unknown")
    make_score = Column(Float, default=0.0)
    make_alternatives = Column(Text, nullable=True)  # JSON
    
    model_value = Column(String(64), default="Unknown")
    model_score = Column(Float, default=0.0)
    model_candidates = Column(Text, nullable=True)  # JSON
    
    plate_text = Column(String(32), default="")
    plate_score = Column(Float, default=0.0)
    
    quality_score = Column(Float, default=0.0)
    model_version = Column(String(64), default="v1.0")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    case = relationship("CaseRecord", back_populates="observations")


class VehicleFusion(Base):
    __tablename__ = "vehicle_attribute_fusion"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(String(64), nullable=False, index=True)
    
    fused_type = Column(String(64), default="unknown")
    fused_type_score = Column(Float, default=0.0)
    
    fused_colour = Column(String(64), default="unknown")
    fused_colour_score = Column(Float, default=0.0)
    colour_alternatives = Column(Text, nullable=True)  # JSON
    
    fused_make = Column(String(64), default="Unknown")
    fused_make_score = Column(Float, default=0.0)
    make_alternatives = Column(Text, nullable=True)  # JSON
    
    fused_model = Column(String(64), default="Unknown")
    fused_model_score = Column(Float, default=0.0)
    model_candidates = Column(Text, nullable=True)  # JSON
    
    fused_plate = Column(String(32), default="")
    plate_conf = Column(Float, default=0.0)
    plate_status = Column(String(32), default="clean")  # clean, partial, unknown
    unknown_positions = Column(Text, nullable=True)  # JSON list [5]
    
    fusion_status = Column(String(32), default="predicted")  # predicted, low_confidence, unknown, conflict, human_confirmed
    frames_used = Column(Integer, default=1)
    timestamps_ms = Column(Text, nullable=True)  # JSON list
    
    best_vehicle_crop_path = Column(String(512), nullable=True)
    best_plate_crop_path = Column(String(512), nullable=True)
    
    model_versions = Column(String(128), default="v1.0")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    case = relationship("CaseRecord", back_populates="fusions")


class InvestigatorCorrection(Base):
    __tablename__ = "investigator_attribute_corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(String(64), nullable=False, index=True)
    attribute_name = Column(String(64), nullable=False)  # vehicle_type, colour, make, model, plate
    original_prediction = Column(String(255), nullable=True)
    corrected_value = Column(String(255), nullable=False)
    investigator_id = Column(String(64), default="INV-001")
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    case = relationship("CaseRecord", back_populates="corrections")


class VehicleRegistry(Base):
    __tablename__ = "vehicle_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    registration_number = Column(String(32), unique=True, index=True, nullable=False)
    vehicle_type = Column(String(64), nullable=False)
    colour = Column(String(64), nullable=False)
    make = Column(String(64), nullable=False)
    model = Column(String(64), nullable=False)
    year = Column(Integer, default=2020)
    owner_masked = Column(String(128), default="* * * * * *")
    state = Column(String(32), default="DL")
    rto = Column(String(64), default="Central")
    registered_at = Column(String(32), default="2020-01-01")
