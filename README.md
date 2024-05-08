# LangChain-Playground

This will be my playground for LangChain

*** EVERYTHING WILL BE RUN LOCALLY ***

*** mamba activate langchain3 ***

## Wednesday, May 8, 2024

Today I needed to install some more libraries to accomodate some notebooks I was running locally in the repo [LLama3_Playground](https://github.com/rkaunismaa/Llama-3-Playground)

39) pip install groq
40) pip install matplotlib
41) pip install langchain-groq

## Tuesday, May 7, 2024

Trying to run [Local RAG agent with LLama3](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_rag_agent_llama3_local.ipynb)

How am I expected to trust the results if the calling the same function 4 times produces 4 different results?!? ... 

38) pip install gpt4all

## Saturday, May 4, 2024

Exploring the function calling abilities of [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)

35) mamba install conda-forge::bitsandbytes

Dammit! Again, after running the conda install of bitsandbytes and then importing, we get the message ...

        Could not find the bitsandbytes CUDA binary at PosixPath('/home/rob/miniforge3/envs/langchain3/lib/python3.11/site-packages/bitsandbytes/libbitsandbytes_cuda118.so')
        The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.

36) mamba remove bitsandbytes
37) pip install bitsandbytes

Yup! That message no longer appears upon import of bitsandbytes.

Wow! Reading from [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) reveals ...

        When quantized versions of the model are released, I recommend using LM Studio for chatting with Hermes 2 Pro. It does not support function calling - for that use our github repo. It is a GUI application that utilizes GGUF models with a llama.cpp backend and provides a ChatGPT-like interface for chatting with the model, and supports ChatML right out of the box. 

So to play with function calling, I cannot use LMStudio! ... I did not know that! ... Good to know, so will play with function calling in jupyter notebooks loading Hermes using HuggingFace transformers.

* Function_Calling/Hermes-2-Pro-Llama-3-8B.ipynb

38) pip install yfinance
39) pip install art

        python functioncall.py --query "I need the current stock price of Tesla (TSLA)"

OK! Nice! I got the above function call to actually work! I terminalled into the ~/Data/Documents/Github/NousResearch/Hermes-Function-Calling$ 
 folder, activated the lanchain3 conda environment, then ran that command. It worked! 

        2024-05-04:19:12:37,957 INFO     [functioncall.py:156] Assistant Message:
        The current stock price of Tesla (TSLA) is $181.19. Let me find more information about Tesla's fundamentals, financial statements, analyst recommendations, and other relevant data. 

        In the next iteration, I'll begin with getting the fundamental data for TSLA.

Hmm ran it again, but this time it did not work ...

        (langchain3) rob@KAUWITB:~/Data/Documents/Github/NousResearch/Hermes-Function-Calling$ python functioncall.py --query "I need the current stock price of Tesla (TSLA)"
        /home/rob/miniforge3/envs/langchain3/lib/python3.11/site-packages/transformers/utils/hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
        warnings.warn(
                                                                                                                dP       
                                                                                                                88       
        88d888b. .d8888b. dP    dP .d8888b. 88d888b. .d8888b. .d8888b. .d8888b. .d8888b. 88d888b. .d8888b. 88d888b. 
        88'  `88 88'  `88 88    88 Y8ooooo. 88'  `88 88ooood8 Y8ooooo. 88ooood8 88'  `88 88'  `88 88'  `"" 88'  `88 
        88    88 88.  .88 88.  .88       88 88       88.  ...       88 88.  ... 88.  .88 88       88.  ... 88    88 
        dP    dP `88888P' `88888P' `88888P' dP       `88888P' `88888P' `88888P' `88888P8 dP       `88888P' dP    dP 
                                                                                                                        
                                                                                                                        

        2024-05-04:19:16:17,866 INFO     [functioncall.py:26] None
        Downloading shards: 100%|██████████████████████| 4/4 [00:00<00:00, 14488.10it/s]
        2024-05-04:19:16:19,221 INFO     [modeling.py:987] We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
        Loading checkpoint shards: 100%|██████████████████| 4/4 [00:04<00:00,  1.19s/it]
        Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
        2024-05-04:19:16:24,803 INFO     [functioncall.py:54] LlamaConfig {
        "_name_or_path": "NousResearch/Hermes-2-Pro-Llama-3-8B",
        "architectures": [
        "LlamaForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128003,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": null,
        "rope_theta": 500000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "float16",
        "transformers_version": "4.40.1",
        "use_cache": false,
        "vocab_size": 128288
        }

        2024-05-04:19:16:24,803 INFO     [functioncall.py:55] GenerationConfig {
        "bos_token_id": 128000,
        "do_sample": true,
        "eos_token_id": 128003
        }

        2024-05-04:19:16:24,804 INFO     [functioncall.py:56] {'bos_token': '<|begin_of_text|>', 'eos_token': '<|im_end|>', 'pad_token': '<|im_end|>'}
        The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
        Setting `pad_token_id` to `eos_token_id`:128003 for open-end generation.
        2024-05-04:19:16:25,190 WARNING  [functioncall.py:72] Assistant message is None
        2024-05-04:19:16:25,190 ERROR    [functioncall.py:161] Exception occurred: Assistant message is None
        Traceback (most recent call last):
        File "/home/rob/Data/Documents/Github/NousResearch/Hermes-Function-Calling/functioncall.py", line 186, in <module>
        inference.generate_function_call(args.query, args.chat_template, args.num_fewshot, args.max_depth)
        File "/home/rob/Data/Documents/Github/NousResearch/Hermes-Function-Calling/functioncall.py", line 162, in generate_function_call
        raise e
        File "/home/rob/Data/Documents/Github/NousResearch/Hermes-Function-Calling/functioncall.py", line 158, in generate_function_call
        recursive_loop(prompt, completion, depth)
        File "/home/rob/Data/Documents/Github/NousResearch/Hermes-Function-Calling/functioncall.py", line 114, in recursive_loop
        tool_calls, assistant_message, error_message = self.process_completion_and_validate(completion, chat_template)
                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/rob/Data/Documents/Github/NousResearch/Hermes-Function-Calling/functioncall.py", line 73, in process_completion_and_validate
        raise ValueError("Assistant message is None")
        ValueError: Assistant message is None

Gonna look into this later ... 


## Friday, May 3, 2024

More Sam Witteveen goodness ... [Adding RAG to LangGraph Agents](https://www.youtube.com/watch?v=WyIWaopiUEo)

* Sam_Witteveen/YT_Adding_RAG_to_LangGraph_Agents_WestWorld.ipynb

34) pip install langchain-chroma

Looks like there is another new model that excels with function calling called [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B).

I have downloaded a quantized version of this [NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF) to LMStudio and am going to explore its performance. You can explore the function calling capabilities at [Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)

Dammit! The actual model I downloaded was 'https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-8B-F16.gguf', and it DOES NOT fully load to the 4090 GPU! Ugh, so instead I am now downloading 'https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf' which I am certain will fit. Yup! It fits!

## Wednesday, May 1, 2024

Starting to look into [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling). I will use the local model "TheBloke/NexusRaven-V2-13B-GGUF/nexusraven-v2-13b.Q8_0.gguf" served up by LMStudio for this. The code samples will reside in the 'Function_Calling' folder.

The details of this model can be found at [Nexusflow/NexusRaven-V2-13B](https://huggingface.co/Nexusflow/NexusRaven-V2-13B)

Also going through [How to use functions with a knowledge base](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_for_knowledge_retrieval.ipynb)

32) mamba install conda-forge::termcolor
33) mamba install conda-forge::arxiv

## Tuesday, April 30, 2024

Working through the Sam Witteveen video tutorial [Creating an AI Agent with LangGraph Llama 3 & Groq](https://www.youtube.com/watch?v=lvQ96Ssesfk), and yes, I will not be using OpenAI or Groq but attempt to run everything locally.

This video is really about using langgraph, which I am looking at for the first time here.

31) pip install langgraph

## Monday, April 29, 2024

Working through 'langchain-ai/rag-from-scratch/rag_from_scratch_15_and_18.ipynb'

30) pip install cohere

        ...
        Installing collected packages: httpx-sse, fastavro, tokenizers, cohere
        Attempting uninstall: tokenizers
        Found existing installation: tokenizers 0.15.2
        Uninstalling tokenizers-0.15.2:
        Successfully uninstalled tokenizers-0.15.2
        ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
        chromadb 0.4.24 requires onnxruntime>=1.14.1, which is not installed.
        transformers 4.39.3 requires tokenizers<0.19,>=0.14, but you have tokenizers 0.19.1 which is incompatible.


ffs! installing cohere broke other packages!! ... this notebook is now complaining about sentence-transformers, so let's re-run the install ... 

31) mamba install conda-forge::sentence-transformers

Nope! Still broken ... complaining about 'tokenizers' ... so lets run ...

32) mamba install conda-forge::tokenizers

OK Nice! That did fix the problem! The notebook now runs locally!

## Sunday, April 28, 2024

Working through 'langchain-ai/rag-from-scratch/rag_from_scratch_12_and_14.ipynb' 

29) pip install RAGatouille


## Saturday, April 27, 2024

Working through 'langchain-ai/rag-from-scratch/rag_from_scratch_10_and_11.ipynb' 

27) mamba install conda-forge::youtube-transcript-api
28) mamba install conda-forge::pytube

1:47 pm - Just checking with conda forge to see the current version of LangChain is at 0.1.16 which was updated '15 days and 13 hours ago', and looking at pip, 0.1.16 was released on April 11, 2024. I am really reluctant to update to 0.1.16 because I am not sure if it will break anything.

## Tuesday, April 23, 2024

Continuing to use some fine tune of Llama 3.

Exploring various notebooks found at [LangChain-AI](https://github.com/langchain-ai)

## Saturday, April 20, 2024

I will no longer use this repo for playing with LLama 3, but I will keep the existing code here and replicate it into a new 'Llama-3-Playground' repo.

## Friday, April 19, 2024

Continuing to look into LLama3.

Remember, the HuggingFace transformers folder is at: ~/.cache/huggingface/hub

How do I get tool usage working when running against a local model?? Is this a property of the model? Gonna look at the LangChain Discord channel to see if anyone knows ... I also want to know if there is a way to have a look at what gets sent to OpenAI and see if I can figure out what is happening ... there must be a way, right?? But thinking about this for a minute, what gets sent does not differ, but what comes back will .. 

Yeah ... it is definitely a propery of the llm being called.

Doing some testing of Llama 3 from the video [LLaMA 3 Tested!! Yes, It’s REALLY That GREAT](https://www.youtube.com/watch?v=0AaNT7XO41I) by Matthew Berman

26) mamba install conda-forge::pygame

## Thursday, April 18, 2024

Wow! LLama 3 dropped today!! Downloading stuff from this environment and will then back up the 2 models. 

* [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
* [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

25) mamba install conda-forge::accelerate

## Wednesday, April 17, 2024

Again spinning in that 'what is the best, most current langchain tutorial I should be going through?' question ...

Gonna stick with this [Tutorials](https://python.langchain.com/docs/additional_resources/tutorials/) for now ...

Really experiencing how much the llm results can vary from one run to another ... it is really helpful to run the cell multiple times to see this, especially when using a local model!

And even if you are able to run the cell without errors, it is still important you actually look at the results to confirm if the model is working as expected.

Yeah. You need to run the cell multiple times to see the results change ... and understand how the model is working ...

BTW prepending the function call with 'if useOpenAI:' supresses any output, so STOP DOING THAT!.

Gonna create a 'Model_Evaluation' sub-folder that will contain notebooks for various local models I want to test through LMStudio. 

Yeah! I want to see exactly what is being sent from Visual Studio Code to LMStudio ... remember Fiddler?? Hmm current searches on 'https debugging proxy' brings up WireShark ... I remember starting to look at this and going nowhere with it ...

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

