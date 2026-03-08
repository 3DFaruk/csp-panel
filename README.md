# Calculation and Optimization Panel (1D Cutting Stock Problem)

This project provides a simple and user-friendly web interface for 1D cutting stock optimization. It helps to calculate the most efficient way to cut linear materials (like pipes, profiles, or wood bars) into smaller pieces while minimizing waste.

## Website

[https://csppanel.streamlit.app/](https://csppanel.streamlit.app/)

## Features

- **Data Entry:** Enter part lengths and required quantities easily through an interactive table.
- **Optimization Algorithms:** 
  - **Advanced Calculation:** Utilizes the `PuLP` library to find the optimal cutting pattern (Integer Linear Programming).
  - **Quick Calculation:** Uses the First-Fit Decreasing algorithm for fast and practical results.
- **Visual PDF Reports:** Generates visual cut lists and patterns and exports them as downloadable PDF files for production.
- **Import/Export:** Supports importing part lists from CSV/Excel and exporting calculated results.
- **Multi-language Support:** The user interface is available in Turkish, English, and Russian.

## AI Assistance

This project was developed with the assistance of Artificial Intelligence in several key areas:
- **Optimization Algorithms:** AI helped implement and refine the Linear Programming (`PuLP`) and First-Fit Decreasing algorithms for the cutting stock problem.
- **PDF Generation:** Creating the visual PDF layouts using `reportlab`, specifically calculating the coordinates, drawing alternating labels, block representations, and checkboxes.
- **Multi-language Architecture:** AI supported the design and integration of the translation system, enabling dynamic switching between Turkish, English, and Russian languages without reloading the app manually.
- **Streamlit UI/UX:** AI helped structure the dashboard layout, integrate interactive components (`st.data_editor`), handling file uploads/downloads, and managing the session state logic.
