"""
Database Session and Initialization
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.app.core.config import settings
from backend.app.database.models import Base, VehicleRegistry

log = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    registry_file = settings.base_dir / "backend" / "data" / "vehicle_registry.json"
    if not registry_file.exists():
        registry_file = settings.base_dir / "backend" / "app" / "database" / "vehicle_registry.json"

    with SessionLocal() as db:
        if registry_file.exists():
            import json
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

                existing_count = db.query(VehicleRegistry).count()
                if existing_count < len(records):
                    db.query(VehicleRegistry).delete()
                    to_add = []
                    for r in records:
                        to_add.append(VehicleRegistry(
                            id=r.get("id"),
                            registration_number=r.get("registration_number"),
                            vehicle_type=r.get("category_type", "car"),
                            colour=r.get("body_colour", "unknown"),
                            make=r.get("manufacturer", "Unknown"),
                            model=r.get("vehicle_model", "Unknown"),
                            year=2021,
                            state=r.get("state", "DL"),
                            rto=r.get("state", "DL"),
                            owner_masked="* * * * * *"
                        ))
                    db.add_all(to_add)
                    db.commit()
                    log.info(f"Loaded {len(to_add)} authorized vehicle records into registry database from {registry_file.name}.")
            except Exception as e:
                log.warning(f"Could not load vehicle_registry.json: {e}")
