import sys
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def ensure_build_deps():
    try:
        import Cython
        import numpy
    except ImportError:
        import subprocess

        pip_executable = sys.executable.replace("python", "pip")
        subprocess.check_call([pip_executable, "install", "cython", "numpy"])


class RDFBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        ensure_build_deps()
        from Cython.Build import cythonize
        from setuptools import Extension
        import numpy

        extensions = [
            Extension(
                "watermark_suite.schemes.rdf.optimized_levenshtein",
                ["src/watermark_suite/schemes/rdf/optimized_levenshtein.pyx"],
                include_dirs=[numpy.get_include()],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
            ),
            Extension(
                "watermark_suite.schemes.rdf.optimized_mersenne",
                ["src/watermark_suite/schemes/rdf/optimized_mersenne.pyx"],
            ),
        ]

        build_data["ext_modules"] = cythonize(
            extensions, compiler_directives={"language_level": "3"}
        )
