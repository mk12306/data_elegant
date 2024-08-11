from setuptools import setup, find_packages

setup(
    name='data_elegant',
    version='0.2',
    description='data_elegant is a data processing module that covers a variety of commonly used data processing methods, aiming to help users process data elegantly',
    author='mk12306',
    author_email='wangkai_12306@163.com',
    packages=['data_elegant'],
    url='',
    install_requires=[
        'numpy==2.0.1',
        'jieba==0.42.1',
        'opencc==1.1.8',
        'sentencepiece==0.2.0',
        'emoji==2.2.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 根据需要调整 Python 版本
    zip_safe=False
)
