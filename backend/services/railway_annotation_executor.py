"""Compatibility imports for the renamed Stage 2 annotation coordinator."""

from backend.services.annotation_coordinator import (
    AnnotationCoordinator,
    annotation_coordinator,
)

# Preserve old imports without keeping Railway-specific execution logic.
RailwayAnnotationExecutor = AnnotationCoordinator
railway_annotation_executor = annotation_coordinator

__all__ = [
    "AnnotationCoordinator",
    "annotation_coordinator",
    "RailwayAnnotationExecutor",
    "railway_annotation_executor",
]
