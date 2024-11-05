# LangChain-Playground

This will be my playground for LangChain

*** EVERYTHING WILL BE RUN LOCALLY ***

*** mamba activate langchain3 ***

## Monday, May 27, 2024

Continuing to look into [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3). 

<strike>mamba activate langchain3</strike>

* pip install mistral_inference ... ran this against langchain3 and it wants to change a lot of stuff, so I bailed on this. 

Instead, I will clone the conda environment ftllm that has pytorch 2.3.0, and use this. 

 1) mamba create --name mistral --clone ftllm
 2) mamba activate mistral
 3) pip install mistral_inference

 mamba activate mistral

 Downloaded a few more 'Mistral-7B-Instruct-v0.3' notebooks into the 'Model_Evaluation/mistralai/Mistral-7B-Instruct-v0.3 ' folder. Yeah, initially nothing langchain related, but will get there once I have stepped through these notebooks. 

 4) pip install mistralai

 5) mamba install conda-forge::scikit-learn
 6) pip install matplotlib ('mamba install conda-forge::matplotlib' would have changed a lot of libraries, so I did not run this.)
 

## Sunday, May 26, 2024

[mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) dropped 4 days ago, and supports function calling. Let's download this model and play ...

## Saturday, May 18, 2024

Downloaded the updated version of the notebook 'How_to_call_functions_with_chat_models.ipynb' to 'How_to_call_functions_with_chat_models_(gpt-4o).ipynb'

## Friday, May 17, 2024

Gonna investigate [Tool Calling with LangChain](https://www.youtube.com/watch?v=zCwuAlpQKTM)

Actually, gonna run through 'How_to_call_functions_for_knowledge_retrieval.ipynb' cuz I don't think I ever did.

## Thursday, May 9, 2024

Working through the code referenced in the Sam Witteveen video [Function Calling with Local Models & LangChain - Ollama, Llama3 & Phi-3](https://www.youtube.com/watch?v=Ss_GdU0KqE0). The code was replicated into the notebook 'Sam_Witteveen/Function_Calling_with_Local_Models.ipynb'

42) pip install langchain-experimental==0.0.57

Installing the current release of langchain-experimental (0.0.58) would have installed a newer version of langchain, so I installed langchain-experimental==0.0.57 instead.

I cloned langchain3 to langchain4, then uninstalled/installed langchain_experimental in langchain4 to get this notebook to work. 

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

## Wednesday, October 30, 2024

Yup, getting back to this. Running stuff from the kitchen computer with the 2070 Super card. Today I created a new langchain conda environment for this. 

 1) mamba create -n langchain python=3.12
 2) mamba activate langchain
 3) mamba install conda-forge::langchain
 4) mamba install conda-forge::jupyterlab

 Going to start with [Build a Simple LLM Application with LCEL](https://python.langchain.com/docs/tutorials/llm_chain/). Ugh the documentation is still lacking with examples on how to run a local large language model. So I am going to go with using LMStudio and the OpenAI API. And I recall a lot of the notebooks in this repository attempt to demonstrate using both the OpenAI models and local models, with the OpenAI models always working better than any local model.

 Here is the current versions of the langchain environment:

        (langchain) rob@rob-MS-7C91:~/Data/Documents/Github/rkaunismaa/LangChain-Playground$ mamba list lang
        # packages in environment at /home/rob/miniforge3/envs/langchain:
        #
        # Name                    Version                   Build  Channel
        langchain                 0.3.5              pyhd8ed1ab_0    conda-forge
        langchain-core            0.3.13             pyhd8ed1ab_0    conda-forge
        langchain-text-splitters  0.3.1              pyhd8ed1ab_0    conda-forge
        langsmith                 0.1.129            pyhd8ed1ab_0    conda-forge

Hmm looking at some of the current notebooks ... where is Mistral these days? When was the last time they put out a new model?

Hmm no LangChain account? Did I sign up with it through github??
Dammit! Trying to sign up with my yahoo email but no confirmation email is ever sent! WTF?!

Jeeze. Signed up through LangSmith ... sigh. 

 5) mamba install conda-forge::openai
 6) mamba install conda-forge::langchain-openai
 7) pip install "langserve[all]"

Nice! Got some of the new stuff to work locally against a model (hermes-3-llama-3.1-8b) served up by LMStudio.

## Friday, November 1, 2024

 8) mamba install conda-forge::langgraph

 Wow. Some LangChain stuff does not work if it does not recognize the model! Running stuff through LMStudio breaks some code but only if you pass in an unrecognized model name, such as "hermes-3-llama-3.1-8b".

  9) mamba install conda-forge::langchain-chroma

Running the above would have changed numerous installed packages ...

        Downgrade:
        ───────────────────────────────────────────────────────────────────────────────────────────────────────────

        - importlib-metadata                              8.5.0  pyha770c72_0            conda-forge     Cached
        + importlib-metadata                              7.0.2  pyha770c72_0            conda-forge       27kB
        - langchain-core                                 0.3.15  pyhd8ed1ab_0            conda-forge     Cached
        + langchain-core                                 0.2.40  pyhd8ed1ab_0            conda-forge      260kB
        - importlib_metadata                              8.5.0  hd8ed1ab_0              conda-forge     Cached
        + importlib_metadata                              7.0.2  hd8ed1ab_0              conda-forge        9kB
        - langgraph                                      0.2.43  pyhd8ed1ab_0            conda-forge     Cached
        + langgraph                                      0.2.40  pyhd8ed1ab_0            conda-forge       88kB
        - langchain-text-splitters                        0.3.1  pyhd8ed1ab_0            conda-forge     Cached
        + langchain-text-splitters                        0.2.4  pyhd8ed1ab_0            conda-forge       27kB
        - langchain-openai                                0.2.4  pyhd8ed1ab_0            conda-forge     Cached
        + langchain-openai                               0.1.25  pyhd8ed1ab_0            conda-forge       38kB
        - langchain                                       0.3.5  pyhd8ed1ab_0            conda-forge     Cached
        + langchain                                      0.2.16  pyhd8ed1ab_0            conda-forge      429kB

Hmm so as a workaround, I am gonna create a new 'langchain-chroma' environment from the current langchain environment, and run the install against that.

 1) mamba create -n langchain-chroma --clone langchain
 2) mamba activate langchain-chroma
 3) mamba install conda-forge::langchain-chroma

 Hmm running the above installs CUDA to the langchain-chroma environment.

 4) mamba install conda-forge::curl

Hmmm gonna attempt running the code found in this [Local RAG Chatbot](https://github.com/grasool/Local-RAG-Chatbot) repository. I will implement this in a 'Local RAG Chatbot' folder, using the langchain-chroma environment.

 5) pip install langchain-community

 Oh FFS! Running the above broke the langchain-chroma environment!! 😠

 
        Installing collected packages: mypy-extensions, marshmallow, typing-inspect, pydantic-settings, dataclasses-json, langchain-core, langchain-text-splitters, langchain, langchain-community
        Attempting uninstall: langchain-core
        Found existing installation: langchain-core 0.2.40
        Uninstalling langchain-core-0.2.40:
        Successfully uninstalled langchain-core-0.2.40
        Attempting uninstall: langchain-text-splitters
        Found existing installation: langchain-text-splitters 0.2.4
        Uninstalling langchain-text-splitters-0.2.4:
        Successfully uninstalled langchain-text-splitters-0.2.4
        Attempting uninstall: langchain
        Found existing installation: langchain 0.2.16
        Uninstalling langchain-0.2.16:
        Successfully uninstalled langchain-0.2.16
        ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
        langchain-openai 0.1.25 requires langchain-core<0.3.0,>=0.2.40, but you have langchain-core 0.3.15 which is incompatible.
        Successfully installed dataclasses-json-0.6.7 langchain-0.3.6 langchain-community-0.3.4 langchain-core-0.3.15 langchain-text-splitters-0.3.1 marshmallow-3.23.1 mypy-extensions-1.0.0 pydantic-settings-2.6.1 typing-inspect-0.9.0

Hmm gonna continue just to see what happens ...

 6) mamba install conda-forge::sentence-transformers ... ugh, did NOT run this because it wants to install the cpu version of pytorch!! Hmm I am gonna first run the install for pytorch ... (current version of cuda in 'langchain-chroma' is 11.8 ... )

 6) mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

 (Yeah, remembering one of the reasons I bailed on LangChain was because of these version conflicts!!)

OK. NOW I am gonna install senetence-transformers ...

 7) mamba install conda-forge::sentence-transformers
 8) mamba install conda-forge::ipywidgets

 Whelp, right now I am thinking of starting over with langchain-chroma, but this time starting with an older version of langchain ... but which?? Dammit, yeah, gonna torch the current langchain-chroma environment and start over with langchain-chroma-v0.2.16 ...

  1) mamba remove -n langchain-chroma --all
  2) mamba create -n langchain-chroma python=3.11
  3) mamba activate langchain-chroma
  4) mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  5) mamba install conda-forge::sentence-transformers
  6) mamba install conda-forge::jupyterlab
  7) mamba install conda-forge::ipywidgets

OK. So at this point, there is no langchain anything installed. So what version should I install??

        (langchain-chroma) rob@rob-MS-7C91:~$ mamba install conda-forge::langchain==0.2.4

        Looking for: ['conda-forge::langchain==0.2.4']

        conda-forge/noarch                                  17.1MB @   3.1MB/s  5.6s
        conda-forge/noarch                                  17.1MB @   2.6MB/s  6.7s
        conda-forge/linux-64                                39.4MB @   3.7MB/s 10.9s
        conda-forge/linux-64                                39.4MB @   3.3MB/s 12.2s

        Pinned packages:
        - python 3.11.*


        Transaction

        Prefix: /home/rob/miniforge3/envs/langchain-chroma

        Updating specs:

        - conda-forge::langchain==0.2.4
        - ca-certificates
        - certifi
        - openssl


        Package                       Version  Build              Channel           Size
        ────────────────────────────────────────────────────────────────────────────────────
        Install:
        ────────────────────────────────────────────────────────────────────────────────────

        + click                         8.1.7  unix_pyh707e725_0  conda-forge     Cached
        + async-timeout                 4.0.3  pyhd8ed1ab_0       conda-forge     Cached
        + tenacity                      8.5.0  pyhd8ed1ab_0       conda-forge     Cached
        + jsonpatch                      1.33  pyhd8ed1ab_0       conda-forge     Cached
        + pyarrow-hotfix                  0.6  pyhd8ed1ab_0       conda-forge       14kB
        + marshmallow                  3.23.0  pyhd8ed1ab_0       conda-forge       93kB
        + mypy_extensions               1.0.0  pyha770c72_0       conda-forge       10kB
        + annotated-types               0.7.0  pyhd8ed1ab_0       conda-forge     Cached
        + nltk                          3.9.1  pyhd8ed1ab_0       conda-forge        1MB
        + typing_inspect                0.9.0  pyhd8ed1ab_0       conda-forge       15kB
        + dataclasses-json              0.6.7  pyhd8ed1ab_0       conda-forge       30kB
        + greenlet                      3.1.1  py311hfdbb021_0    conda-forge      240kB
        + libsentencepiece              0.2.0  hc20e799_5         conda-forge      823kB
        + pydantic-core                2.23.4  py311h9e33e62_0    conda-forge        2MB
        + orjson                      3.10.10  py311h9e33e62_0    conda-forge      310kB
        + sqlalchemy                   2.0.36  py311h9ecbd09_0    conda-forge        4MB
        + sentencepiece-spm             0.2.0  hc20e799_5         conda-forge       87kB
        + sentencepiece-python          0.2.0  py311h5c6ee89_5    conda-forge        2MB
        + sentencepiece                 0.2.0  h38be061_5         conda-forge       17kB
        + pydantic                      2.9.2  pyhd8ed1ab_0       conda-forge     Cached
        + langsmith                   0.1.129  pyhd8ed1ab_0       conda-forge     Cached
        + langchain-core               0.2.40  pyhd8ed1ab_0       conda-forge     Cached
        + langchain-text-splitters      0.2.4  pyhd8ed1ab_0       conda-forge     Cached
        + langchain                     0.2.4  pyhd8ed1ab_0       conda-forge      416kB

        Upgrade:
        ────────────────────────────────────────────────────────────────────────────────────

        - datasets                     2.14.4  pyhd8ed1ab_0       conda-forge     Cached
        + datasets                     2.21.0  pyhd8ed1ab_0       conda-forge      363kB

        Downgrade:
        ────────────────────────────────────────────────────────────────────────────────────

        - fsspec                    2024.10.0  pyhff2d567_0       conda-forge     Cached
        + fsspec                     2024.5.0  pyhff2d567_0       conda-forge      216kB
        - numpy                         2.1.2  py311h71ddf71_0    conda-forge     Cached
        + numpy                        1.26.4  py311h64a7726_0    conda-forge        8MB
        - sentence-transformers         3.2.1  pyhd8ed1ab_0       conda-forge     Cached
        + sentence-transformers         2.7.0  pyhd8ed1ab_0       conda-forge      109kB

        Summary:

        Install: 24 packages
        Upgrade: 1 packages
        Downgrade: 3 packages

        Total download: 19MB

        ────────────────────────────────────────────────────────────────────────────────────


        Confirm changes: [Y/n] n

Nope! Not gonna install that ... what about a higher version??


        (langchain-chroma) rob@rob-MS-7C91:~$ mamba install conda-forge::langchain==0.2.16

        Looking for: ['conda-forge::langchain==0.2.16']

        conda-forge/linux-64                                        Using cache
        conda-forge/noarch                                          Using cache
        conda-forge/linux-64                                        Using cache
        conda-forge/noarch                                          Using cache

        Pinned packages:
        - python 3.11.*


        Transaction

        Prefix: /home/rob/miniforge3/envs/langchain-chroma

        Updating specs:

        - conda-forge::langchain==0.2.16
        - ca-certificates
        - certifi
        - openssl


        Package                     Version  Build            Channel           Size
        ────────────────────────────────────────────────────────────────────────────────
        Install:
        ────────────────────────────────────────────────────────────────────────────────

        + greenlet                    3.1.1  py311hfdbb021_0  conda-forge      240kB
        + pydantic-core              2.23.4  py311h9e33e62_0  conda-forge        2MB
        + orjson                    3.10.10  py311h9e33e62_0  conda-forge      310kB
        + sqlalchemy                 2.0.36  py311h9ecbd09_0  conda-forge        4MB
        + async-timeout               4.0.3  pyhd8ed1ab_0     conda-forge     Cached
        + tenacity                    8.5.0  pyhd8ed1ab_0     conda-forge     Cached
        + annotated-types             0.7.0  pyhd8ed1ab_0     conda-forge     Cached
        + jsonpatch                    1.33  pyhd8ed1ab_0     conda-forge     Cached
        + pydantic                    2.9.2  pyhd8ed1ab_0     conda-forge     Cached
        + langsmith                 0.1.129  pyhd8ed1ab_0     conda-forge     Cached
        + langchain-core             0.2.40  pyhd8ed1ab_0     conda-forge     Cached
        + langchain-text-splitters    0.2.4  pyhd8ed1ab_0     conda-forge     Cached
        + langchain                  0.2.16  pyhd8ed1ab_0     conda-forge     Cached

        Downgrade:
        ────────────────────────────────────────────────────────────────────────────────

        - numpy                       2.1.2  py311h71ddf71_0  conda-forge     Cached
        + numpy                      1.26.4  py311h64a7726_0  conda-forge        8MB

        Summary:

        Install: 13 packages
        Downgrade: 1 packages

        Total download: 14MB

        ────────────────────────────────────────────────────────────────────────────────


        Confirm changes: [Y/n] 

Hmm yeah, going with 0.2.16 ....

 8) mamba install conda-forge::langchain==0.2.16
 9) mamba install conda-forge::openai

Now let's continue with the other langchain packages ...

10) mamba install conda-forge::langchain-chroma ...NOPE! This version 0.1.4 would change a ton of stuff! ..
10) mamba install conda-forge::langchain-chroma=0.1.3 ... wow! Same with this version ... keep going ...
10) mamba install conda-forge::langchain-chroma=0.1.2 ... still nope! All there want to downgrade a lot of libraries ... sigh.

This makes me think why not install langchain-chroma FIRST before any other langchain stuff to see what it wants to install? ...

Yeah. Let's try that, shall we ... 

  1) mamba remove -n langchain-chroma --all
  2) mamba create -n langchain-chroma python=3.11
  3) mamba activate langchain-chroma
  4) mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  5) mamba install conda-forge::sentence-transformers
  6) mamba install conda-forge::jupyterlab
  7) mamba install conda-forge::ipywidgets

OK. So now lets try to install langchain-chroma and see what we get ... 


        (langchain-chroma) rob@rob-MS-7C91:~$ mamba install conda-forge::langchain-chroma

        Looking for: ['conda-forge::langchain-chroma']

        conda-forge/linux-64                                        Using cache
        conda-forge/noarch                                          Using cache
        conda-forge/linux-64                                        Using cache
        conda-forge/noarch                                          Using cache

        Pinned packages:
        - python 3.11.*


        Transaction

        Prefix: /home/rob/miniforge3/envs/langchain-chroma

        Updating specs:

        - conda-forge::langchain-chroma
        - ca-certificates
        - certifi
        - openssl


        Package                                      Version  Build                 Channel           Size
        ──────────────────────────────────────────────────────────────────────────────────────────────────────
        Install:
        ──────────────────────────────────────────────────────────────────────────────────────────────────────

        + bcrypt                                       4.2.0  py311h9e33e62_1       conda-forge      248kB
        + httptools                                    0.6.1  py311h9ecbd09_1       conda-forge       91kB
        + watchfiles                                  0.24.0  py311h9e33e62_1       conda-forge      396kB
        + websockets                                    13.1  py311h9ecbd09_0       conda-forge      237kB
        + pydantic-core                               2.23.4  py311h9e33e62_0       conda-forge     Cached
        + cryptography                                43.0.3  py311hafd3f86_0       conda-forge        1MB
        + wrapt                                       1.16.0  py311h9ecbd09_1       conda-forge       63kB
        + libuv                                       1.49.2  hb9d3cd8_0            conda-forge     Cached
        + orjson                                     3.10.10  py311h9e33e62_0       conda-forge     Cached
        + chroma-hnswlib                               0.7.3  py311h5510f57_2       conda-forge      180kB
        + uvloop                                      0.21.0  py311h9ecbd09_1       conda-forge      677kB
        + libpulsar                                    3.5.1  hc08aa7d_2            conda-forge     Cached
        + protobuf                                    4.25.3  py311hbffca5d_1       conda-forge      399kB
        + pulsar-client                                3.5.0  py311hfdbb021_1       conda-forge      356kB
        + grpcio                                      1.62.2  py311ha6695c7_0       conda-forge        1MB
        + python-multipart                            0.0.17  pyhff2d567_0          conda-forge     Cached
        + starlette                                   0.41.2  pyha770c72_0          conda-forge     Cached
        + dnspython                                    2.7.0  pyhff2d567_0          conda-forge     Cached
        + importlib-resources                          6.4.5  pyhd8ed1ab_0          conda-forge     Cached
        + pypika                                      0.48.9  pyhd8ed1ab_0          conda-forge     Cached
        + tenacity                                     8.5.0  pyhd8ed1ab_0          conda-forge     Cached
        + python-dotenv                                1.0.1  pyhd8ed1ab_0          conda-forge     Cached
        + click                                        8.1.7  unix_pyh707e725_0     conda-forge     Cached
        + durationpy                                     0.9  pyhd8ed1ab_0          conda-forge     Cached
        + annotated-types                              0.7.0  pyhd8ed1ab_0          conda-forge     Cached
        + backoff                                      2.2.1  pyhd8ed1ab_0          conda-forge     Cached
        + monotonic                                      1.5  pyhd8ed1ab_1          conda-forge     Cached
        + opentelemetry-semantic-conventions          0.37b0  pyhd8ed1ab_0          conda-forge       29kB
        + jsonpatch                                     1.33  pyhd8ed1ab_0          conda-forge     Cached
        + cachetools                                   5.5.0  pyhd8ed1ab_0          conda-forge     Cached
        + pyu2f                                        0.1.5  pyhd8ed1ab_0          conda-forge     Cached
        + pyasn1                                       0.6.1  pyhd8ed1ab_1          conda-forge     Cached
        + pyjwt                                        2.9.0  pyhd8ed1ab_1          conda-forge     Cached
        + blinker                                      1.8.2  pyhd8ed1ab_0          conda-forge     Cached
        + python-flatbuffers                         24.3.25  pyh59ac667_0          conda-forge     Cached
        + shellingham                                  1.5.4  pyhd8ed1ab_0          conda-forge     Cached
        + mdurl                                        0.1.2  pyhd8ed1ab_0          conda-forge     Cached
        + humanfriendly                                 10.0  pyhd8ed1ab_6          conda-forge     Cached
        + pyopenssl                                   24.2.1  pyhd8ed1ab_2          conda-forge     Cached
        + deprecated                                  1.2.14  pyh1a96a4e_0          conda-forge     Cached
        + opentelemetry-proto                         1.16.0  pyhd8ed1ab_0          conda-forge       38kB
        + googleapis-common-protos                    1.65.0  pyhd8ed1ab_0          conda-forge     Cached
        + email-validator                              2.2.0  pyhd8ed1ab_0          conda-forge     Cached
        + typer-slim                                  0.12.5  pyhd8ed1ab_0          conda-forge     Cached
        + uvicorn                                     0.32.0  pyh31011fe_1          conda-forge     Cached
        + pydantic                                     2.9.2  pyhd8ed1ab_0          conda-forge     Cached
        + posthog                                      3.6.5  pyhd8ed1ab_0          conda-forge     Cached
        + pyasn1-modules                               0.4.1  pyhd8ed1ab_0          conda-forge     Cached
        + rsa                                            4.9  pyhd8ed1ab_0          conda-forge     Cached
        + oauthlib                                     3.2.2  pyhd8ed1ab_0          conda-forge     Cached
        + markdown-it-py                               3.0.0  pyhd8ed1ab_0          conda-forge     Cached
        + coloredlogs                                 15.0.1  pyhd8ed1ab_3          conda-forge     Cached
        + opentelemetry-api                           1.16.0  pyhd8ed1ab_0          conda-forge       40kB
        + email_validator                              2.2.0  hd8ed1ab_0            conda-forge     Cached
        + uvicorn-standard                            0.32.0  h31011fe_1            conda-forge     Cached
        + langsmith                                  0.1.129  pyhd8ed1ab_0          conda-forge     Cached
        + google-auth                                 2.35.0  pyhff2d567_0          conda-forge     Cached
        + requests-oauthlib                            2.0.0  pyhd8ed1ab_0          conda-forge     Cached
        + rich                                        13.9.3  pyhd8ed1ab_0          conda-forge     Cached
        + opentelemetry-sdk                           1.16.0  pyhd8ed1ab_0          conda-forge       63kB
        + langchain-core                              0.2.40  pyhd8ed1ab_0          conda-forge     Cached
        + python-kubernetes                           31.0.0  pyhd8ed1ab_0          conda-forge     Cached
        + typer-slim-standard                         0.12.5  hd8ed1ab_0            conda-forge     Cached
        + opentelemetry-exporter-otlp-proto-grpc      1.16.0  pyhd8ed1ab_0          conda-forge       21kB
        + typer                                       0.12.5  pyhd8ed1ab_0          conda-forge     Cached
        + fastapi-cli                                  0.0.5  pyhd8ed1ab_1          conda-forge     Cached
        + fastapi                                    0.115.4  pyhd8ed1ab_0          conda-forge     Cached
        + onnxruntime                                 1.19.2  py311h9b445dc_0_cpu   conda-forge       12MB
        + chromadb                                    0.4.17  py311h38be061_0       conda-forge      772kB
        + langchain-chroma                             0.1.4  pyhd8ed1ab_0          conda-forge     Cached

        Downgrade:
        ──────────────────────────────────────────────────────────────────────────────────────────────────────

        - aws-c-common                                0.9.31  hb9d3cd8_0            conda-forge     Cached
        + aws-c-common                                0.9.23  h4ab18f5_0            conda-forge      236kB
        - libabseil                               20240722.0  cxx17_h5888daf_1      conda-forge     Cached
        + libabseil                               20240116.2  cxx17_he02047a_1      conda-forge     Cached
        - libthrift                                   0.21.0  h0e7cc3e_0            conda-forge     Cached
        + libthrift                                   0.19.0  hb90f79a_1            conda-forge      409kB
        - numpy                                        2.1.2  py311h71ddf71_0       conda-forge     Cached
        + numpy                                       1.26.4  py311h64a7726_0       conda-forge     Cached
        - s2n                                          1.5.5  h3931f03_0            conda-forge     Cached
        + s2n                                         1.4.16  he19d79f_0            conda-forge      350kB
        - aws-checksums                               0.1.20  hf20e7d7_3            conda-forge     Cached
        + aws-checksums                               0.1.18  he027950_7            conda-forge       50kB
        - aws-c-sdkutils                              0.1.19  hf20e7d7_6            conda-forge     Cached
        + aws-c-sdkutils                              0.1.16  he027950_3            conda-forge       55kB
        - aws-c-compression                           0.2.19  hf20e7d7_4            conda-forge     Cached
        + aws-c-compression                           0.2.18  he027950_7            conda-forge       19kB
        - aws-c-cal                                    0.7.4  hd3f4568_4            conda-forge     Cached
        + aws-c-cal                                   0.6.15  h816f305_1            conda-forge       47kB
        - libre2-11                               2024.07.02  hbbce691_1            conda-forge     Cached
        + libre2-11                               2023.09.01  h5a48ba9_2            conda-forge     Cached
        - libprotobuf                                 5.27.5  h5b01275_2            conda-forge     Cached
        + libprotobuf                                 4.25.3  hd5b35b9_1            conda-forge     Cached
        - aws-c-io                                   0.14.20  h389d861_2            conda-forge     Cached
        + aws-c-io                                    0.14.9  hd3d3696_3            conda-forge      159kB
        - re2                                     2024.07.02  h77b4e00_1            conda-forge     Cached
        + re2                                     2023.09.01  h7f4b329_2            conda-forge     Cached
        - orc                                          2.0.2  h690cf93_1            conda-forge     Cached
        + orc                                          2.0.1  h17fec99_1            conda-forge        1MB
        - aws-c-http                                  0.8.10  h6bb76cc_5            conda-forge     Cached
        + aws-c-http                                   0.8.2  h75ac8c9_3            conda-forge      195kB
        - aws-c-event-stream                           0.5.0  h72d8268_0            conda-forge     Cached
        + aws-c-event-stream                           0.4.2  hb72ac1a_14           conda-forge       54kB
        - libgrpc                                     1.65.5  hf5c653b_0            conda-forge     Cached
        + libgrpc                                     1.62.2  h15f2491_0            conda-forge     Cached
        - aws-c-mqtt                                  0.10.7  had056f2_5            conda-forge     Cached
        + aws-c-mqtt                                  0.10.4  hb0abfc5_7            conda-forge      163kB
        - aws-c-auth                                  0.7.31  hcdce11a_5            conda-forge     Cached
        + aws-c-auth                                  0.7.22  hf36ad8f_6            conda-forge      106kB
        - libgoogle-cloud                             2.30.0  h438788a_0            conda-forge     Cached
        + libgoogle-cloud                             2.25.0  h2736e30_0            conda-forge        1MB
        - aws-c-s3                                     0.7.0  hc85afc5_0            conda-forge     Cached
        + aws-c-s3                                    0.5.10  h44b787d_4            conda-forge      110kB
        - libgoogle-cloud-storage                     2.30.0  h0121fbd_0            conda-forge     Cached
        + libgoogle-cloud-storage                     2.25.0  h3d9a0c8_0            conda-forge      761kB
        - aws-crt-cpp                                 0.29.0  h07ed512_0            conda-forge     Cached
        + aws-crt-cpp                                0.26.12  he940a02_1            conda-forge      340kB
        - aws-sdk-cpp                               1.11.407  h9c41b47_6            conda-forge     Cached
        + aws-sdk-cpp                               1.11.329  h0f5bab0_6            conda-forge        4MB
        - libarrow                                    18.0.0  h9c5d0aa_0_cuda       conda-forge     Cached
        + libarrow                                    16.1.0  h9102155_9_cpu        conda-forge        8MB
        - pyarrow-core                                18.0.0  py311hcae7c52_0_cuda  conda-forge     Cached
        + pyarrow-core                                16.1.0  py311h8c3dac4_4_cpu   conda-forge        4MB
        - libparquet                                  18.0.0  hdbc8f64_0_cuda       conda-forge     Cached
        + libparquet                                  16.1.0  h6a7eafb_9_cpu        conda-forge        1MB
        - libarrow-acero                              18.0.0  h530483c_0_cuda       conda-forge     Cached
        + libarrow-acero                              16.1.0  hac33072_9_cpu        conda-forge      600kB
        - libarrow-dataset                            18.0.0  h530483c_0_cuda       conda-forge     Cached
        + libarrow-dataset                            16.1.0  hac33072_9_cpu        conda-forge      581kB
        - libarrow-substrait                          18.0.0  h8ffff87_0_cuda       conda-forge     Cached
        + libarrow-substrait                          16.1.0  h7e0c224_9_cpu        conda-forge      549kB
        - pyarrow                                     18.0.0  py311hbd00459_0       conda-forge     Cached
        + pyarrow                                     16.1.0  py311hbd00459_4       conda-forge       28kB

        Summary:

        Install: 70 packages
        Downgrade: 31 packages

        Total download: 43MB

        ──────────────────────────────────────────────────────────────────────────────────────────────────────


        Confirm changes: [Y/n] 

Interesting! So all those downgrades are still there! They are not because of an existing install of langchain! And this does not even install langchain!! ... I was expecting it to ALSO install langchain because it has not yet been installed! Jeeze!!

OK. So yeah, let's install langchain ... 

 8) mamba install conda-forge::langchain==0.2.16
 9) mamba install conda-forge::openai

Now let's try langchain-openai ... 

10) mamba install conda-forge::langchain-openai=0.1.25
11) pip install langchain-community==0.2.17

OK. So up to this point, there have been no breaking changes or downgrades. Now let's try langchain-chroma ...

12) mamba install conda-forge::langchain-chroma

This install langchain-chroma 0.1.4, which is the latest version. 30 Packages were downgraded to accomodate this install.

## Monday, November 4, 2024

Let's get back to our initial langchain environment.

mamba activate langchain

 9) pip install langchain-anthropic
10) pip install langchain-community==0.2.17

FFS ... I ran the above install and it changed ... 

        Installing collected packages: mypy-extensions, marshmallow, typing-inspect, dataclasses-json, langchain-core, langchain-text-splitters, langchain, langchain-community
        Attempting uninstall: langchain-core
        Found existing installation: langchain-core 0.3.15
        Uninstalling langchain-core-0.3.15:
        Successfully uninstalled langchain-core-0.3.15
        Attempting uninstall: langchain-text-splitters
        Found existing installation: langchain-text-splitters 0.3.1
        Uninstalling langchain-text-splitters-0.3.1:
        Successfully uninstalled langchain-text-splitters-0.3.1
        Attempting uninstall: langchain
        Found existing installation: langchain 0.3.5
        Uninstalling langchain-0.3.5:
        Successfully uninstalled langchain-0.3.5
        ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
        langserve 0.3.0 requires langchain-core<0.4,>=0.3, but you have langchain-core 0.2.43 which is incompatible.
        langchain-openai 0.2.4 requires langchain-core<0.4.0,>=0.3.13, but you have langchain-core 0.2.43 which is incompatible.
        langchain-anthropic 0.2.4 requires langchain-core<0.4.0,>=0.3.15, but you have langchain-core 0.2.43 which is incompatible.
        Successfully installed dataclasses-json-0.6.7 langchain-0.2.16 langchain-community-0.2.17 langchain-core-0.2.43 langchain-text-splitters-0.2.4 marshmallow-3.23.1 mypy-extensions-1.0.0 typing-inspect-0.9.0

Dammit ... I bet stuff is now broken! Idiot! Sigh, gonna rebuild the langchain environment ... 

  1) mamba remove -n langchain --all
  2) mamba create -n langchain python=3.11
  3) mamba activate langchain
  4) mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  5) mamba install conda-forge::sentence-transformers
  6) mamba install conda-forge::jupyterlab
  7) mamba install conda-forge::ipywidgets

Now back to the 'what version of langchain should I install??' question .... 

  8) mamba install conda-forge::langchain==0.2.16
  9) mamba install conda-forge::openai
 10) mamba install conda-forge::langchain-openai==0.1.25
 11) pip install langchain-community==0.2.17
 12) pip install langchain-anthropic==0.1.23
 13) pip install langgraph==0.2.40

 These langchain versions were selected to ensure langchain-core was not changed.

 Hmm I am thinking of exploring other paid llm api's other than OpenAI. Had a look at Anthropic, but they only provide a fixed monthly cost, so on thanks. I want something like OpenAI, pay by usage, so if I don't hit it for a month, there is no money wasted.
 
 Signed up with Mistral.ai. Checking out their payment plans. Nice! They have a free tier! Hmm, the MistralAI code sample still uses OpenAI for embeddings! 

 Hmm looks like we need langchain_huggingface for the langchain-chroma environment. 

 mamba activate langchain-chroma
 13) pip install langchain-huggingface==0.0.3

 We also want to install to the langchain environment ...

 mamba activate langchain
  14) pip install langchain-huggingface==0.0.3

  Looks like we some more stuff installed to the langchain environment ...

   15) mamba install conda-forge::sqlite

   16) mamba install conda-forge::faiss-cpu

   Installing faiss-gpu would have changed a lot of libraries, so I didn't run it.

   FFS! Something is now broken in the langchain environment, but seems to be working in the langchain-chroma environment ... 

   ## Tuesday, November 5, 2024

   Gonna rebuild the langchain environment because it is now broken ... sigh ... 

  1) mamba remove -n langchain --all
  2) mamba create -n langchain python=3.11
  3) mamba activate langchain
  4) mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  5) mamba install conda-forge::sentence-transformers
  6) mamba install conda-forge::jupyterlab
  7) mamba install conda-forge::ipywidgets
  8) mamba install conda-forge::langchain==0.2.16
  9) mamba install conda-forge::openai
 10) mamba install conda-forge::langchain-openai==0.1.25
 11) pip install langchain-community==0.2.17
 12) pip install langchain-anthropic==0.1.23
 13) pip install langgraph==0.2.40
 14) pip install langchain-huggingface==0.0.3

 OK! Nice! Now the langchain environment no longer throws that error in the 'LangChain Tutorials/LMStudio/Build a Question Answering system over SQL data.ipynb' notebook.

So again, noticing, how different local models behave differently than OpenAI models, and some langchain code samples fail for this reason.

Hmm need to install langchain-ollama to the langchain-chroma environment.

mamba activate langchain-chroma

 14) pip install langchain-ollama==0.1.3

 Now I need pypdf for the langchain environment.

 mamba activate langchain

  15) mamba install conda-forge::pypdf




