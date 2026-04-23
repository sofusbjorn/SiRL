# Simulation-based Reinforcement Learning Course Note

This hosts the repository for building the course note contents as website or PDF through [jupyter-book](https://jupyterbook.org/en/stable/intro.html). So far the finding is that using [MyST Markdown](https://mystmd.org/guide/quickstart-jupyter-lab-myst) and relevant addons are sufficiently featured to create interactive technical tutorials/books. Some of them have been explored for the interests of SiRL contents:

* Latex syntax for writing math equations;
* citing/referencing bibtex literature, equations and figures;
* embedding/using jupyter lab/notebook to show code example and execution result (generated as static pages);
* interactive plot for some libraries with input from jupyter widgets;
* excuting code of the book/website on a local jupyter lab server (might be able to edit as well but not explored yet);
* rendering in web page with pythreejs and mediapy;

Known issue so far:
* Using native myst for dynamic running of code cells works but it seems not to support global numbering of equations so far. Now equation numbering will reset per page. Cross-reference might look confusing even the links are pointing to the correct item.

## For Users/Developers
The course notes can be compiled and visualized as web pages hosted on a local machine. It may also be browsed and run as separated jupyter notebook/lab pages.

It is recommended to create venv or conda environment for the notes as it contain its own code and dependencies.

Install the local python code:
```
pip install -e src/python
```
Install the required jupyter and myst dependencies for building the notes:
```
pip install -r requirements.txt
```

To run the server and website, follow the standard mystmd process:
```
myst start
```

or together with a jupyter server if you wanted to run the code over the web page, see [instructions](https://mystmd.org/guide/execute-notebooks) or use a script like below:
```
# Set local environment variable
port="8888"

# Setup environment variables used by MyST
export JUPYTER_BASE_URL="http://localhost:${port}"
export JUPYTER_TOKEN="sirl"

#shutdonw the server when ctrl-c is pressed to stop running MyST
sigterm_handler() { 
  echo "Attempting to stop jupyter server..."
  jupyter server stop
  exit 1
}

trap 'trap " " SIGINT SIGTERM SIGHUP; kill 0; wait; sigterm_handler' SIGINT SIGTERM SIGHUP

# Start server in the background
jupyter lab --IdentityProvider.token="${JUPYTER_TOKEN}" --ServerApp.port="${port}" --NotebookApp.allow_origin='http://localhost:3000'    &

# Run MyST
myst start --execute
```

## For deployment on a remote host
TBD: docker on ucloud?