![image](https://github.com/patternizer/quantum_poetry/blob/master/title_frame.jpg)

# Transdisciplinary Quantum Poetry

## [World Lines: A Quantum Supercomputer Poem](https://www.amycatanzano.com/world-lines)

A poem by [Amy Catanzano](https://www.amycatanzano.com) based on a theoretical model of a topological quantum computer

Formats: print publication (complete), computational poetry and interactive digital poetry (underway), 3D art installation (anticipated)

Collaborator for Phase 3: Dr. Michael Taylor, applied mathematician and senior research associate in climate science at the University of East Anglia (Norwich, United Kingdom)

Description: World Lines: A Quantum Supercomputer Poem is a poem and poetic form invented by Amy Catanzano that is based on a theoretical model of a topological quantum computer. Phase 1 of the project is complete and was published by the Simons Center for Geometry and Physics at Stony Brook University. Additional poems by Amy Catanzano using this poetic form are underway in Phase 2.

In Phase 3 of the project, underway, Michael Taylor is using the Python computer programming language and machine learning (artificial intelligence) to develop an algorithm and quantum script that computationally expresses all possible versions of World Lines. After parsing each sentence in the poem and identifying branch points, words that are in common, Dr. Taylor is training a linguistic processor to choose world lines that are semantically logical to track how different topological paths move through a text map into different versions of the poem. A web interface will be generated where, after a text is loaded, a World Lines algorithm could find the branch points and do one of two things: 1) allow the reader to manually navigate along a world line, creating a new poem as a re-structured sample of the text that could be stored and studied, and 2) run a simulation and generate world lines that the reader could choose between in order to render new poems. Visual poetry and artwork are being generated from the data.

Phase 4 of the poem will involve creating a 3D art installation based on the poem.

Anticipated outcomes for Phases 3-4: computational poetry, visual poetry and artwork, evolution of quantum script writing, interactive web interface, investigation of quantum linguistics and information theory, educational tool in poetry and physics, 3D art installation.

This is the codebase for an interactive app being developed and tested and which is deployed here: http://quantum-poetry.herokuapp.com/

## Contents

* `quantum_poetry.py` - main script to be run with Python 3.6+
* `app.py` - Plotly Dash interactive app script to be run with Python 3.6+

The first step is to clone the latest quantum_poetry code and step into the check out directory: 

    $ git clone https://github.com/patternizer/quantum_poetry.git
    $ cd quantum_poetry
    
### Using Standard Python 

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested 
in a conda virtual environment running a 64-bit version of Python 3.6+.

quantum_poetry can be run from sources directly, once the modules in requirements.txt are resolved.


Run with:

    $ python quantum_poetry.py
	
A static version of the app can be run locally with:

    $ python app.py
	
        
## License

The code is distributed under terms and conditions of the [MIT license](https://opensource.org/licenses/MIT).

## Contact information

* [Michael Taylor](https://patternizer.github.io)


