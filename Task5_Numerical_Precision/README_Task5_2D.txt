TASK 5 - NUMERICAL PRECISION (2D)
================================

Contents
--------
Task5_Numerical_Precision_Report_2D.pdf   Final report
Task5_Numerical_Precision_Report_2D.docx  Editable report
task5_precision_2d.py                     Reproducible 2D experiment script
requirements.txt                           Python dependencies
task5_2d_results/                         CSV measurements, plots, environment details

Model used
----------
This Task 5 experiment follows the quantitative 2D model in the team's
uploaded task3_4.py file:
N=12, L=10.0, k=15.0, r0=1.5, rc=1.5, dt=0.005, T=10.0, seed=42.
It uses gamma=0 and Velocity-Verlet for the precision comparison.

Note: The separate simulator.py animation file uses a different 2D soft-
repulsion / hard-core-collision variant. Do not mix its force parameters with
the task3_4.py energy/integrator model when reporting Task 5.

Run on Windows / VS Code
------------------------
1. Open the project folder in VS Code.
2. Open Terminal -> New Terminal.
3. Install dependencies:
      python -m pip install -r requirements.txt
4. Run the experiment:
      python task5_precision_2d.py
5. Open task5_2d_results/ for the generated outputs.

Output files
------------
task5_precision_metrics.csv      Summary: memory, median time, energy and position error
task5_timing_samples.csv         All five raw runtime measurements
task5_precision_timeseries.csv   Energy and position-error values at each time step
task5_energy_comparison.png      Kinetic, potential and total energy comparison
task5_position_error.png         Energy drift and relative position difference
task5_runtime_comparison.png     Five runtime samples for each precision
task5_environment.txt            Python, NumPy, Matplotlib and platform details

Interpretation note
-------------------
The report's runtime values are from the environment in which this package
was generated. Runtime is hardware- and environment-dependent. Run the script
on the project computer and use its generated timing results for any final
hardware-specific claim. The float64 trajectory is a numerical reference for
comparison, not an exact analytical solution.
