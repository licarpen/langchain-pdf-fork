from functools import partial
from .chatopenai import build_llm

llm_map = {
    "gpt-3.5-turbo-0613": partial(build_llm, model_name="gpt-3.5-turbo-0613"),
    "gpt-3.5-turbo": partial(build_llm, model_name="gpt-3.5-turbo")
}
