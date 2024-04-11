# LangChain-Playground

This will be my playground for LangChain

*** EVERYTHING WILL BE RUN LOCALLY ***

## Thursday, April 11, 2024

This repository will contain various notebooks and code samples for playing with LangChain. As of today, pip install langchain is at version 0.1.15 and conda-forge is at 0.1.15. 

I will be using the conda environment 'langchain' which has this latest version:

    (langchain) rob@KAUWITB:~$ mamba list chain
    # packages in environment at /home/rob/miniforge3/envs/langchain:
    #
    # Name                    Version                   Build  Channel
    langchain                 0.1.15             pyhd8ed1ab_0    conda-forge
    langchain-community       0.0.32             pyhd8ed1ab_0    conda-forge
    langchain-core            0.1.41             pyhd8ed1ab_0    conda-forge
    langchain-openai          0.0.5                    pypi_0    pypi
    langchain-text-splitters  0.0.1              pyhd8ed1ab_0    conda-forge

This was used for some previous work on LangChain and contains a ton of other libraries.

The impulse right now is to try to do EVERYTHING locally, so NEVER using OpenAI and knowing this will add to the complexity of this project. No doubt, HuggingFace will play an important role is this objective. 

Gawd is it easy to get lost in the weeds as soon as I deviate outside of all the 'use OpenAI' examples I have seen.

And looking at the numerous LangChain tutorials on YouTube quickly reveals most of them are from a year ago, so the thinking is they are outdated and may not work with the latest version of LangChain. So, I think the approach should be to use these existing notebooks as a guide on some sequential learning stream, but using the latest version of LangChain and running everything locally. Yes this will make stuff harder, but that is what I want to do.

[A Complete LangChain Guide](https://nanonets.com/blog/langchain/) gonna start from here, cuz why not ... And the notebooks I create from this guide will be written into the 'nanonets' folder.

Wow ... first kick at langchain fails in the langchain environment, but then works in the langchain2 environment. Hmmm considering creating a new environment for this repository. Yeah, going with this ... 

 1) mamba create -n langchain3 python=3.11
 2) mamba activate langchain3
 3) mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
 4) mamba install conda-forge::jupyterlab
 5) mamba install conda-forge::scikit-learn
 6) mamba install conda-forge::langchain
 7) mamba install conda-forge::openai

 Yup, now it works! So going forward, I will be using the 'langchain3' environment.

