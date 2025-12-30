from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=1,
    max_tokens=4096,
)

model2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Generative Adversarial Networks (or GANs for short) are one of the most popular Machine Learning algorithms developed in recent times.

For those new to the field of Artificial Intelligence (AI), we can briefly describe Machine Learning (ML) as the sub-field of AI that uses data to “teach” a machine/program how to perform a new task. A simple example of this would be using images of a person’s face as input to the algorithm, so that a program learns to recognize that same person in any given picture (it’ll probably need negative samples too). For this purpose, we can describe Machine Learning as applied mathematical optimization, where an algorithm can represent data (e.g. a picture) in a multi-dimensional space (remember the Cartesian Plane? That’s a 2 dimensional field), and then learns to distinguish new multi-dimensional vector samples as belonging to the target distribution or not. For a visual understanding on how machines learn I recommend this broad video explanation and this other video on the rise of machines, which I were very fun to watch. Though this is a very fascinating field to explore and discuss, I’ll leave the in-depth explanation for a later post, we’re here for GANs!

Press enter or click to view image in full size

Google Trend’s Interest over time for term “Generative Adversarial Networks”
What’s so magical about GANs?
In short, they belong to the set of algorithms named generative models. These algorithms belong to the field of unsupervised learning, a sub-set of ML which aims to study algorithms that learn the underlying structure of the given data, without specifying a target value. Generative models learn the intrinsic distribution function of the input data p(x) (or p(x,y) if there are multiple targets/classes in the dataset), allowing them to generate both synthetic inputs x’ and outputs/targets y’, typically given some hidden parameters.

In contrast, supervised learning algorithms learn to map a function y’=f(x), given labeled data y. An example of this would be classification, where one could use customer purchase data (x) and the customer respective age (y) to classify new customers. Most of the supervised learning algorithms are inherently discriminative, which means they learn how to model the conditional probability distribution function (p.d.f) p(y|x) instead, which is the probability of a target (age=35) given an input (purchase=milk). Despite the fact that one could make predictions with this probability distribution function, one is not allowed to sample new instances (simulate customers with ages) from the input distribution directly.
Side-note: It is possible to use discriminative algorithms which are not probabilistic, they are called discriminative functions.

GANs they have proven to be really succesfull in modeling and generating high dimensional data, which is why they’ve become so popular. Nevertheless they are not the only types of Generative Models, others include Variational Autoencoders (VAEs) and pixelCNN/pixelRNN and real NVP. Each model has its own tradeoffs.

Some of the most relevant GAN pros and cons for the are:

They currently generate the sharpest images
They are easy to train (since no statistical inference is required), and only back-propogation is needed to obtain gradients
GANs are difficult to optimize due to unstable training dynamics.
No statistical inference can be done with them (except here):
GANs belong to the class of direct implicit density models; they model p(x) without explicitly defining the p.d.f.
So.. why generative models?
Generative models are one of the most promising approaches to understand the vast amount of data that surrounds us nowadays. According to OpenAI, algorithms which are able to create data might be substantially better at understanding intrinsically the world. The idea that generative models hold a better potential at solving our problems can be illustrated using the quote of one of my favourite physicists.

“What I cannot create, I do not understand.” — Richard P. Feynman

(I strongly suggest reading his book “Surely You’re Joking Mr. Feynman”)

Generative models can be thought as containing more information than their discriminative counterpart/complement, since they also be used for discriminative tasks such as classification or regression (where the target is a continuous value such as ℝ). One could calculate the conditional p.d.f p(y|x) needed most of the times for such tasks, by using statistical inference on the joint p.d.f. p(x,y) if it is available in the generative model.

Though generative models work for classification and regression, fully discriminative approaches are usually more successful at discriminative tasks in comparison to generative approaches in some scenarios.

Use Cases
Among several use cases, generative models may be applied to:

Generating realistic artwork samples (video/image/audio).
Simulation and planning using time-series data.
Statistical inference.
Machine Learning Engineers and Scientists reading this article may have already realized that generative models can also be used to generate inputs which may expand small datasets.
I also found a very long and interesting curated list of awesome GAN applications here.

2. Understanding a GAN: Overview
Press enter or click to view image in full size

Global concept of a GAN
Generative Adversarial Networks are composed of two models:

The first model is called a Generator and it aims to generate new data similar to the expected one. The Generator could be asimilated to a human art forger, which creates fake works of art.
The second model is named the Discriminator. This model’s goal is to recognize if an input data is ‘real’ — belongs to the original dataset — or if it is ‘fake’ — generated by a forger. In this scenario, a Discriminator is analogous to an art expert, which tries to detect artworks as truthful or fraud.
How do these models interact? Paraphrasing the original paper which proposed this framework, it can be thought of the Generator as having an adversary, the Discriminator. The Generator (forger) needs to learn how to create data in such a way that the Discriminator isn’t able to distinguish it as fake anymore. The competition between these two teams is what improves their knowledge, until the Generator succeeds in creating realistic data.

Mathematically Modeling a GAN
Though the GANs framework could be applied to any two models that perform the tasks described above, it is easier to understand when using universal approximators such as artificial neural networks.

A neural network G(z, θ₁) is used to model the Generator mentioned above. It’s role is mapping input noise variables z to the desired data space x (say images). Conversely, a second neural network D(x, θ₂) models the discriminator and outputs the probability that the data came from the real dataset, in the range (0,1). In both cases, θᵢ represents the weights or parameters that define each neural network.

As a result, the Discriminator is trained to correctly classify the input data as either real or fake. This means it’s weights are updated as to maximize the probability that any real data input x is classified as belonging to the real dataset, while minimizing the probability that any fake image is classified as belonging to the real dataset. In more technical terms, the loss/error function used maximizes the function D(x), and it also minimizes D(G(z)).

Furthermore, the Generator is trained to fool the Discriminator by generating data as realistic as possible, which means that the Generator’s weight’s are optimized to maximize the probability that any fake image is classified as belonging to the real dataset. Formally this means that the loss/error function used for this network maximizes D(G(z)).

In practice, the logarithm of the probability (e.g. log D(…)) is used in the loss functions instead of the raw probabilies, since using a log loss heavily penalises classifiers that are confident about an incorrect classification.

Press enter or click to view image in full size

Log Loss Visualization: Low probability values are highly penalized
After several steps of training, if the Generator and Discriminator have enough capacity (if the networks can approximate the objective functions), they will reach a point at which both cannot improve anymore. At this point, the generator generates realistic synthetic data, and the discriminator is unable to differentiate between the two types of input.

Since during training both the Discriminator and Generator are trying to optimize opposite loss functions, they can be thought of two agents playing a minimax game with value function V(G,D). In this minimax game, the generator is trying to maximize it’s probability of having it’s outputs recognized as real, while the discriminator is trying to minimize this same value.

Press enter or click to view image in full size

"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()

 