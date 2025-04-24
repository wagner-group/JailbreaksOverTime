import os

os.system(
    "cat data/oai_moderation_00 data/oai_moderation_01 > data/oai_moderation.dill"
)

from setuptools import find_packages, setup


def read(file_name):
    with open(
        os.path.join(os.path.dirname(__file__), file_name), encoding="utf-8"
    ) as f:
        return f.read()


def get_requirements():
    requirements = read("requirements.txt")
    return [r.strip() for r in requirements.split("\n") if r.strip()]


setup(
    name="llmad",
    version="1.0",
    url="https://github.com/julien-piet/llm-attack-detect",
    description="Query LlamaGuard model",
    long_description=read("README.md"),
    packages=find_packages("src", exclude=["tests*"]),
    include_package_data=True,
    package_dir={"": "src"},
    install_requires=get_requirements(),
    entry_points={
        "console_scripts": [
            "llmad=llmad.pipeline.run:main",
            "llmad_score=llmad.pipeline.score_jailbreaks:main",
            "train_cli=llmad.finetune.train:train_cli",
        ],
    },
)
