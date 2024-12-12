%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Simulation files
Miguel Parada Contzen
Conce, December 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The files contained in this folder are for computing the examples in the paper

	Parada Contzen, Miguel. 'Power sharing and voltage regulation control for grid forming of future distribution circuits with multiple voltage references', 2024.

All the simulation figures in the paper are obtained using conventional desktop hardware and the following Python 3.10 scripts:

	- Example_paper.py: Main file for closed and open loop simulation of described system and power control algorithm. It produces all simulation figures in the paper.
	- circuit_fun.py: Functions used for defining arbitrary circuits, in terms of topology and parameters values.
	- loads_lib.py: Functions for defining arbitrary load profiles based on domiciliary behavior including different load devices and the charging of electric vehicles
	- time_fun.py: Functions for interpoling the dynamic behavior of photo-voltaic and load profiles. And Newton-Raphson algorithms for power flow calculations.
	- pv_profile.csv: photo-voltaic profiles at an example location.
