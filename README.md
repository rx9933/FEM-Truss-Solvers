<h1 align="center">FEM Truss Solver</h1>
<p align="center">
  <b> :Linear and Non-linear deformation solver for truss problems (external force or deformation constrains). </b></br>
  <sub><sub>
</p>
<br />
    
# Program Files
1) "linear_method.py" is used for linear (small deformation problems). Note that for large deformations, the solver cannot approximate/converge to an accurate answer.
2) "eulers_method.py" uses the euler method to approximate non-linear deformations.
3) "newton_raphson.py" uses the newton-raphson method to approximate non-linear deformations (is most acurate out of the 3 methods).

The .txt files represent the input data files (different truss structures and constraints. For any of the methods/code files, ensure to change the "dataname" variable to the appropriate problem/data file before running the program.
Use 'Animated_Truss.blend' after running the program to visualize the stress/strain/deformations of the truss structure:
[![IMAGE ALT TEXT](http://img.youtube.com/vi/3hXeJQ47j-w/0.jpg)](https://youtu.be/3hXeJQ47j-w?si=v_Oy51CwoIDUegq9 "Animated Truss Visualization")
