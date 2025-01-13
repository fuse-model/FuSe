import setuptools

setuptools.setup(
    name='widowx_envs',
    version='0.0.1',
    packages=setuptools.find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
        entry_points={
        'console_scripts': [
            'widowx_env_service = widowx_envs.widowx_env_service:main',
        ],
    },
)
