# Training of the beta-VAE

## Configuration
First, you need to update `config.py` with:
- your directories
- your input data dimensions
```
self.data_dir = "/path/to/data/directory"
self.subject_dir = "/path/to/list_of_subjects"
self.save_dir = "/path/to/saving/directory"

self.in_shape = (c, h, w, d)
```

## Running the model
You can run the model using:
``` bash
$ python3 main.py
```
