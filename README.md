# BANG 
## BAyesian decomposiotioN of Galaxies

BANG is a GPU/CPU-python code for modelling both the photometry and kinematics of galaxies.
The underlying model is the superposition of different component the user has 3 possible 
combination:

* Bulge + inner disc + outer disc + Halo
* Bulge +  disc  + Halo
* inner disc + outer disc + Halo

For any detail about the model construction see Rigamonti et al. 2022.

The parameter estimation is done with a python implementation [CPnest](https://github.com/johnveitch/cpnest) 
of nested sampling algorithm.

We strongly suggest to run BANG on GPU. CPU parameter estimation can take
days. A fast CPU implementation will be available in a future release of the code.


All the function needed by the user are well documented. In order to run BANG on 
your galaxy open the example.py script and follow the instructions.

Once your data have been correctly prepared and the config.yaml file has been compiled, 
running BANG requires few lines of code.


For any problem or suggestion feel free to contact the authors at:

            frigamonti@uninsubria.it

