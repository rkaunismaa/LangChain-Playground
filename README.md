# LangChain-Playground

This will be my playground for LangChain

*** EVERYTHING WILL BE RUN LOCALLY ***

*** mamba activate langchain3 ***

## Sunday, April 14, 2024

Starting to go through the [LangChain v0.1.0](https://www.youtube.com/playlist?list=PLfaIDFEXuae0gBSJ9T0w7cu7iJZbH3T31) videos on YouTube by Harrison Chase. This content is from January 2024 ...

mamba activate langchain3

22) mamba install conda-forge::langchain-openai
23) pip install langchainhub
24) pip install tavily-python

## Saturday, April 13, 2024

Today I was running some of the code in 'Greg_Kamradt/5_Levels_of_Text_Splitting.ipynb' against OpenAI because I wanted to see what gets returned. 

## Friday, April 12, 2024

15) mamba install conda-forge::chromadb
16) mamba install conda-forge::faiss

Noticed when faiss was installed, it looks like the cuda version of faiss was installed, which is great ... !

        Package          Version  Build                        Channel         Size
        ───────────────────────────────────────────────────────────────────────────────
        Install:
        ───────────────────────────────────────────────────────────────────────────────

        + libfaiss         1.7.4  cuda112hb18a002_0_cuda       conda-forge     71MB
        + libfaiss-avx2    1.7.4  cuda112h1234567_0_cuda       conda-forge     71MB
        + faiss            1.7.4  py311cuda112h9171b99_0_cuda  conda-forge      3MB

        Summary:

        Install: 3 packages

        Total download: 145MB

Ugh ... some of the code provided in [A Complete LangChain Guide](https://nanonets.com/blog/langchain/) is wrong and not working. Gonna temporarily deviate to another notebook by Greg Kamradt and come back to this Module 2 - Retrieval.ipynb later ...

Moving onto 'Greg_Kamradt/5_Levels_of_Text_Splitting.ipynb' ...

17) mamba install conda-forge::llama-index

Installing llama-index made these changes to this langchain3 environment ...

        Package                         Version  Build            Channel           Size
        ────────────────────────────────────────────────────────────────────────────────────
        Install:
        ────────────────────────────────────────────────────────────────────────────────────

        + aiostream                       0.5.2  pyhd8ed1ab_0     conda-forge     Cached
        + dirtyjson                       1.0.8  pyhd8ed1ab_0     conda-forge     Cached
        + types-futures                   3.3.8  pyhd8ed1ab_0     conda-forge     Cached
        + types-protobuf        4.25.0.20240410  pyhd8ed1ab_0     conda-forge       59kB
        + llama-index                    0.9.48  pyhc1e730c_0     conda-forge     Cached

        Downgrade:
        ────────────────────────────────────────────────────────────────────────────────────

        - libsentencepiece                0.2.0  hb0b37bd_1       conda-forge     Cached
        + libsentencepiece               0.1.99  hb0b37bd_8       conda-forge      836kB
        - scikit-learn                    1.4.2  py311hc009520_0  conda-forge     Cached
        + scikit-learn                    1.2.2  py311hc009520_2  conda-forge     Cached
        - sentencepiece-spm               0.2.0  hb0b37bd_1       conda-forge     Cached
        + sentencepiece-spm              0.1.99  hb0b37bd_8       conda-forge       88kB
        - sentencepiece-python            0.2.0  py311h99063f3_1  conda-forge     Cached
        + sentencepiece-python           0.1.99  py311h99063f3_8  conda-forge        3MB
        - sentencepiece                   0.2.0  h38be061_1       conda-forge     Cached
        + sentencepiece                  0.1.99  h38be061_8       conda-forge       31kB


18) pip install unstructured
19) pip install pdf2image

... ran into an issue with pdf2image ... gonna run the conda install ...

20) mamba install conda-forge::pdf2image

... nope ... still missing pillow_heif ...

21) pip install pillow_heif

... wow ... now complains about cv2 ...

22) mamba install conda-forge::opencv  ... this would have broken the langchain3 environment ... so bailed ... 

OK. For this 'Greg_Kamradt/5_Levels_of_Text_Splitting.ipynb' notebook, I'm going to switch to the 'mls2' conda environment ... it allows us to take this notebook further than the 'langchain3' environment. 


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

 Here is another tutorial I will walk my way through ...

 [LangChain Python Tutorial: The Ultimate Step-by-Step Guide](https://analyzingalpha.com/langchain-python-tutorial)

 I am noticing in my testing with LMStudio is that the server will just stop working if we hit an error. The only solution is to unload the model, reload and restart the server and then test it with the LMStudio test code. 

 8) mamba install conda-forge::pypdf
 9) mamba install conda-forge::pdfminer
10) mamba install conda-forge::pdfminer.six
11) pip install lxml
12) mamba install conda-forge::tiktoken
13) mamba install conda-forge::sentence-transformers
14) mamba install conda-forge::ipywidgets

