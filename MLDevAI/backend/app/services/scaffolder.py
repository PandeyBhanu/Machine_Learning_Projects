from jinja2 import Environment, FileSystemLoader
import os
import shutil


def scaffold_project(context: dict, template_root: str = "templates") -> str:
    task = context["task"]
    template_path = os.path.join(template_root, task)
    output_dir = "output/project"

    # Clean existing output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    env = Environment(loader=FileSystemLoader([template_path, f"{template_root}/base"]))

    files_to_generate = {
        "train.py": "train.py.jinja",
        "requirements.txt": "requirements.txt.jinja",
        "README.md": "README.md.jinja",
    }

    for filename, template_name in files_to_generate.items():
        template = env.get_template(template_name)
        rendered = template.render(**context)
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(rendered)

    # ZIP the project
    zip_path = shutil.make_archive("output/project", "zip", output_dir)
    return zip_path
