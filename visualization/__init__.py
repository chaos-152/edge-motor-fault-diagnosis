"""Visualization and dashboard package."""
from .plot_signals import generate_all_plots
from .dashboard import generate_html_dashboard

__all__ = ["generate_all_plots", "generate_html_dashboard"]
