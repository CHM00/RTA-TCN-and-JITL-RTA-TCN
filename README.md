# A temporal convolution network-based just-in-time learning method for industrial quality variable prediction  

## RTA-TCN and JITL-RTA-TCN

### Explanation of code
The files JITL-RTA-TCN-DC.py and JITL-RTA-TCN-SRU.py implement a just-in-time learning method, while RTA-TCN-DC.py and RTA-TCN-SRU.py utilize the RTA-TCN method.  
The datasets are provided in Debutanizer_data.txt and SRU_data.txt. Specifically, Debutanizer_data.txt is used by both JITL-RTA-TCN-DC.py and RTA-TCN-DC.py, whereas SRU_data.txt is used by JITL-RTA-TCN-SRU.py and RTA-TCN-SRU.py.  

This code section references the following CSDN articles:
1. [TCN Paper and Code Interpretation Summary](https://blog.csdn.net/qq_33331451/article/details/104810419)
2. [Deep Learning — Attention Scores (Notes + Code)](https://blog.csdn.net/jbkjhji/article/details/128956369)

## Citation Request
If you find our work valuable and use this paper or its findings in your research, publications, or projects, please cite our paper using the following citation information:

    @article{ZHENG2024168,  
    title = {A temporal convolution network-based just-in-time learning method for industrial quality variable prediction},  
    journal = {Chemical Engineering Research and Design},  
    volume = {212},  
    pages = {168-184},  
    year = {2024},  
    issn = {0263-8762},  
    doi = {https://doi.org/10.1016/j.cherd.2024.11.003},  
    url = {https://www.sciencedirect.com/science/article/pii/S0263876224006324},  
    author = {Xiaoqing Zheng and Baofan Wu and Huiming Chen and Anke Xue and Song Zheng and Ming Ge and Yaguang Kong},  
    keywords = {Soft sensor, Just-in-time learning (JITL), Temporal convolutional network (TCN), Industrial process, Quality variables, Temporal attention},  
    abstract = {Real-time acquisition of quality variables is paramount for enhancing control and optimization of industrial processes. Process modeling methods, such as soft sensors, offer a means to predict difficult-to-obtain quality variables using easily measurable process parameters. However, the dynamic nature of industrial processes poses significant challenges to modeling. For instance, conventional models are typically trained offline using historical data, rendering them incapable of adapting to real-time changes in data distribution or environmental conditions. To tackle this challenge, we introduce a novel approach termed the Residual Temporal Attention Temporal Convolution Network (RTA-TCN) and propose a just-in-time learning method based on RTA-TCN for industrial process modeling. The RTA-TCN model incorporates temporal attention into TCN, enabling the integration of previous time-step process variables into the current ones, as well as the fusion of internally relevant features among inputs. Moreover, to prevent the partial loss of original information during feature integration, residual connections are introduced into the temporal attention mechanism. These connections facilitate the retention of original feature information to a maximal extent while integrating relevant features. Consequently, the proposed RTA-TCN demonstrates significant advantages in handling the non-linearity and long-term dynamic dependencies inherent in industrial variables. Additionally, the proposed just-in-time learning method leverages RTA-TCN as a local model and updates it in real-time using online industrial data. This just-in-time learning method enables effective adaptation to varying data distributions and environmental conditions. We validate the performance of our method using two industrial datasets (Debutanizer Column and Sulfur Recovery Unit).}
    }




## How to Cite
Citing works you build upon is crucial for scientific integrity and reproducibility. It allows others to trace the origins of ideas and methods used in various studies. By citing this paper, you help us demonstrate the impact of our research and potentially aid in securing future funding and collaborations.

For more information on why and how to cite sources, consider visiting https://doi.org/10.1016/j.cherd.2024.11.003.

Thank you for acknowledging our contribution by citing our work appropriately!





