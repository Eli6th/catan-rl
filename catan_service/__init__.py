"""Canonical service layer for the Catan engine and web/API surfaces."""

from .flask_app import create_app

__all__ = ["create_app"]
