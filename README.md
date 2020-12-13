My solutions to the assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

Note that I self-studied the course, so I cannot verify my solutions (although based on my results they seem to be correct).
To try my solutions on your own computer, make sure you have `pipenv` installed. I used `pipenv` to create and manage a virtualenv for each homework.
Run `pipenv install` in each individual directory to install the required packages. I ran into an issue installing `torch` with `pipenv`, so in 
addition to running `pipenv install` go to [pytorch.org](https://pytorch.org/) and use their installation guide (make sure package manager is `pip`) to
install `torch`. Also make sure that MuJoCo is installed and in the correct directory. See `installation.md` inside the `hw1` folder for instructions.

All deliverables are in the `soln_pdf` folder inside each individual homework directory (e.g. `hw2/soln_pdf/hw2.pdf`).

This was developed on a Windows 10 PC using Python 3.7. Other versions of python probably work fine too, but I have not tested them.  

**UPDATE 12/12/20:** I've migrated to Ubuntu 18.04. 

For those who still want to work on this on Windows, a few notes:

`pipenv install` (probably) will still work. I've removed the `Pipenv.lock` files from the repository for easier cross-platform migration and support. 
Additionally, official Windows support for `mujoco-py` is deprecated, although it still worked on my system, so your performance may vary. 

If you want to get my code to run on your system, a few installation notes:

I've downgraded to CUDA version 10.2, so now `pipenv install torch torchvision` works again (on Windows I had CUDA 11.0 installed), I recommend
staying at CUDA 10.2 for now since `tensorflow` does not even support CUDA 11 (it also only technically supports 10.1, but according to many people online
it should work perfectly fine for 10.2). If you have some other version of CUDA installed other than 10.2, then you need to use `pip` to install `torch`.
Otherwise, run `pipenv install --python 3.7.x` inside each individual directory, and it should install all necessary packages. Replace the `x` with the latest
version. As of writing this, the latest python 3.7 version was 3.7.9. Also, since `pipenv` is slow, I would also consider `poetry` for package management or 
just not using `pipenv` (or any other python package manager) at all and using `python -m venv` or `virtualenv` along with `pip`.


A few bits about getting `mujoco-py` to install on Ubuntu:

If you run into this error:

```
fatal error: GL/osmesa.h: No such file or directory
```

Run this command:

```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

For more details, see [here](https://github.com/openai/mujoco-py#ubuntu-installtion-troubleshooting).


If you run into this error:

```
No such file or directory: 'patchelf': 'patchelf'
```

Run this command:

```
sudo apt install patchelf
```

The above command worked for me on Ubuntu 18.04, for other versions of Ubuntu you may need to add a PPA. See [this](https://github.com/openai/mujoco-py/issues/147#issuecomment-361417560)
link for more info. 
