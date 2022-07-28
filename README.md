# Point cloud pathdinfing tool

## About The Project

This is a python tool that uses data clustering techniques to identify relevant locations inside Lidar point cloud data.
The output is a JSON file containing targets in the shape of coordinates and positions that the camera must go in order to have a clear view over the targets.

This tool can be useful in application with [this adaptation of the Potree tool](https://github.com/MazieroMarco/bt-visualization-tool) that allows the import of such JSON files and the visualization of camera paths.

### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Python3][Python3]][Python3-url]
* [![Scikitlearn][Scikitlearn]][Scikitlearn-url]
* [![JSON][JSON]][JSON-url]
* [![Numpy][Numpy]][Numpy-url]
* [![laspy][laspy]][laspy-url]


## Getting Started

This project is a very simple python file that can be executed to generate a JSON with the targets and positions.

### Prerequisites

To be sure everything is going to work properly, be sure tou have Python version >= 3.8

* Get python version
  ```sh
  python --version
  ```

### Installation

Then you must install the required libraries to be able to use the script properly.

1. Clone the repo
   ```sh
   git clone https://github.com/MazieroMarco/bt-pathfinding-tool.git
   ```
2. Install python packages
   ```sh
   pip install -r requirements.txt
   ```
3. Run the pathfinder script
   ```sh
   python pathfinder.py my_file.las -o ./my_output.json
   ```

## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

* Run simple clustering with default params (5 targets, 0.1% data sample size)
   ```sh
   python pathfinder.py my_file.las
   ```
   
 * Run simple clustering with custom params (10 targets, 0.4% data sample size, custom epsilon, custom output)
   ```sh
   python pathfinder.py my_file.las -o ./output.json -p 10 -q 0.4
   ```
   
 More details are avilable in the comand line tool help menu by using the `-h` option.
 
 * Help menu
    ```
   python pathfinder.py -h
   usage: pathfinder.py [-h] [--output DIR] [--poi N] [--quantity N] [--epsilon N] INPUT

    Finds interesting locations and a camera path inside a given LAS point cloud data file.

    positional arguments:
      INPUT                 The path of the input LAS data file

    optional arguments:
      -h, --help            show this help message and exit
      --output DIR, -o DIR  The output directory for the generated JSON file
      --poi N, -p N         The amount of points of interest to output
      --quantity N, -q N    The proportion of points to keep in the working data sample [0 < q < 1].
                            Warning, a big number slows down the algorithm.
      --epsilon N, -e N     The epsilon parameter used for the data clustering. This parameter is
                            approximated if no value is given.
   ```
 
<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

Marco Maziero - maziero.marco@outlook.com

<!-- MARKDOWN LINKS & IMAGES -->
[Python3]: https://img.shields.io/badge/Python3.8-35495E?style=for-the-badge&logo=python&logoColor=4FC08D
[Python3-url]: https://www.python.org/
[Scikitlearn]: https://img.shields.io/badge/Scikitlean-DD0031?style=for-the-badge&logo=scikitlearn&logoColor=white
[Scikitlearn-url]: https://scikit-learn.org
[JSON]: https://img.shields.io/badge/JSON-4A4A55?style=for-the-badge&logo=json&logoColor=FF3E00
[JSON-url]: https://www.json.org
[Numpy]: https://img.shields.io/badge/Numpy-0769AD?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org/
[laspy]: https://img.shields.io/badge/Laspy-563D7C?style=for-the-badge&logo=laspy&logoColor=white
[laspy-url]: https://laspy.readthedocs.io/en/latest/

