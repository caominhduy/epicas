import importlib.resources as pkg_resources
import json

def static_params():
    with pkg_resources.open_text('epicas.settings', 'hyperparams.json') as f:
        parameters = json.load(f)

    return parameters
