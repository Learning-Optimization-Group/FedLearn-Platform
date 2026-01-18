from setuptools import setup, find_packages
import os


# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements = []
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')

    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Skip PyTorch lines (users install with CUDA support separately)
                    if not line.startswith('torch') and not line.startswith('torchvision'):
                        requirements.append(line)

    return requirements


# Read long description from README
def read_long_description():
    """Read long description from README.md"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


setup(
    name='fedlearn',
    version='0.1.0',
    description='Distributed Federated Learning Framework with DeComFL Integration',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='Learning Optimization Group',
    author_email='haibo.yang@rit.edu',
    url='https://github.com/Learning-Optimization-Group/FedLearn-Platform',
    license='Apache License 2.0',

    # Package configuration
    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    # Python version requirement
    python_requires='>=3.10',

    # Dependencies
    install_requires=read_requirements(),

    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-asyncio>=0.21.0',
            'pytest-cov>=4.1.0',
            'ruff>=0.6.0',
        ],
        'gpu': [
            # PyTorch with CUDA support (install manually)
            # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ],
    },

    # Entry points for CLI tools (if any)
    entry_points={
        'console_scripts': [
            # Add CLI commands here if needed
            # 'fedlearn-server=fedlearn.server.cli:main',
        ],
    },

    # Classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Keywords
    keywords='federated-learning machine-learning deep-learning distributed-computing pytorch',

    # Include package data
    include_package_data=True,

    # Zip safe
    zip_safe=False,

    # Project URLs
    project_urls={
        'Documentation': 'https://github.com/Learning-Optimization-Group/FedLearn-Platform/tree/main/framework/docs',
        'Source': 'https://github.com/Learning-Optimization-Group/FedLearn-Platform',
        'Bug Reports': 'https://github.com/Learning-Optimization-Group/FedLearn-Platform/issues',
    },
)