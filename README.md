## Read and Save ABAQUS .odb file using pickle

This is a ABAQUS python script for reading a outputs data base file (.odb) and saving the results to a folder of local files.

### Some Illustration:
* For some cases, it could convenient that the output results can be processed in your own external Python environment.
* The scripts published by other developers so far are not efficient and generalized enough, which could become more excessive in batched cases such as surrogate modeling and optimization.
* In this simple script, all the functions are realized only by some built-in Python libraries such as collections, os, and numpy. Thus, you can manipulate the .pkl results easily or extend the functions.
* The entire result data files are saved and constructed in a root folder of the ABAQUS job. Each level of directory will separately contain various types of data such as FieldOutpus and HistoryOutputs. The reason we build the database is that if the data of a case is very large, saving all the data in one file is very inefficient. Furthermore, we do not want any additional library  support in the local ABAQUS environment as it could make the implementation very complicated.
* At the current stage, only some cases are included in the script. Actually we have only tested it on some explicit dynamic analyses of solid mechanics. If you have any other requirement, you may extend your own scripts under this architecture since the structure of a ABAQUS .odb file is fixed and unified as to the best of our knowledge.


### Usage

The script is easy to use for it just create a single python object and a single method.

* 1.Copy the script to the ABAQUS python environment folder, for a typical SMApy directory:
  ```sh
  ..\SIMULIA\EstProducts\2020\win_b64\tools\SMApy\python2.7\Lib\site-packages\
  ```
* 2.Import the pickler class:
  ```sh
  from odb_pickler import AbaqusODBPickler
  ```
* 3.Instantiate a pickler class, then call the method to read and save:
  ```sh
  pickler = AbaqusODBPickler(
   queried_invariants=['MISES','MAGNITUDE'],
    odb_file_path=r"Job-1.odb",
    save_dir=r"Job-1-Results"
   )
   pickler.struct(
        queried_fields = ['UT', 'PEEQ', 'A'],
        queried_invariants = ['MISES','MAGNITUDE'],
        st_frame = 10,
        ed_frame = 30
    )
   pickler.close()

  ```
### Some detailed descriptions


1. Constructor AbaqusODBPickler(                 odb_file_path, save_dir='', odb_read_only=True,                  odb_read_internal_sets=False)

A convertor for read ABAQUS .odb file using pickle

Parameters:
```sh
odb_file_path (str): Path of the .odb file to be read and saved.
save_dir (str, optional): Root folder directory of the pickled files.Defaults to empty, and the pickled files will besaved in the same directory with the odb file.
odb_read_only (bool, optional): Whether to read only the odb file. Defaults to True.
odb_read_internal_sets (bool, optional): Whether to read the internal set of the odb file. Defaults to False.
```     
2. Method AbaqusODBPickler.struct(read_odb_obj=True, read_model_data=True, st_frame=1,             ed_frame=-1, queried_fields='ALL',
               queried_invariants=['MISES', 'MAGNITUDE'])
        
Read and save odb data to the directory of self.save_root_dir

Parameters:
```sh
read_odb_obj (bool, optional): Whether to read the attributes data of the ODB object. Defaults to True.
read_model_data (bool, optional): Whether to read the model data of the odb.rootAssembly object. Defaults to True.
st_frame (int, optional): The starting frame. Defaults to 1.
ed_frame (int, optional): The ending frame. Defaults to -1.
queried_fields (str|sequence[str], optional): The fieldOutput labels to read. Defaults to 'ALL'.
queried_invariants (list, optional): The invariant labels of a fieldOutput to read. Defaults to ['MISES', 'MAGNITUDE'].
  ```    

3. Method AbaqusODBPickler.close()
Close the .odb file. **Make sure to call this function once, after the AbaqusODBPickler reads the data.**


<!-- LICENSE -->
## License


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

My Email - wzm1997@hnu.edu.cn

My Team in the HNU Project Link: [https://github.com/HnuAiSimOpt]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
