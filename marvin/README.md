#Marvin#

Marvin is a GPU-only neural network framework made with simplicity, hackability, speed, memory consumption, and high dimensional data in mind.

#Usage#

1. Install CUDNN v3, CUDA (with CUBLAS)  

2. Run mac.sh or linux.sh

#TODO#


0. Logarithmic softmax layer was added

1. LRNLayer: LRNCrossChannel, LRN and LCN., Divisive Normalization layer

2. alex net need group in Conv layer

3. AlexNet on ImageNet => ImageList data layer, data prefectching using async + cudaMemcpy memory to the GPU with a pointer to the
data to save copy time?
data prefectching + cudaMemcpy memory to the GPU with a pointer to the
data to save copy time

4. extra feature function better (shuran)

4. Loss layer in GPU


4. FP16

5. 3D Shape Net

6. concatenation layer

7. mask layer and weight layer (just like dropout? mask from training data)
support spatial mask and label mask in loss layer.

8. multi-gpu + multi-thread

9. googlenet, vgg net 

10. deconvolution

11. testing time dynamic memory per layer?

12. padding to handle different image resize

13. Implement detection
faster RCNN and YoLo

14. Matlab and python interface

16. c++ curves (SVG)

17. auto detecting of bad lose to automatically re-initialize

18. train layer by layer

19. recurrent neural network

20.easier to others to create new layers
 is it possible that when create a new layer, user just need to
define a forward function, then we numerically compute gradient
required in back propagation? In this way, user doesn't even need to
define backward function.




#About the name#

the depressed robot from Hitchhiker's guide (cute mascot and plenty of small jokes that people who grew up reading science fiction might appreciate) 

Marvin Minsky is 
father of AI
The term artifical intellgence (AI) was first coid by him and John McCarthy.
He built SNARC in 1951, the first randomly wired neural network learning machine, during his PhD at Princeton (where I teach).
Made out of 400 vacuum tubes, the machine was an early attempt at creating a learning system based on neural nets modeled on those the human brain uses.
Minsky wrote the book Perceptrons (with Seymour Papert), which became the foundational work in the analysis of artificial neural networks. 
He is co-founder of MIT's AI lab, where I got my PhD.

Anti-logic: He found that solving difficult problems in vision and NLP required ad-hoc solutions – they argued that there was no simple and general principle (like logic) that would capture all the aspects of intelligent behavior.

