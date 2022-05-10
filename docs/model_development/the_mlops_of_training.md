# The MLOps of Training

Training is a big and meaningful phase in every ML / DL based solution life cycle. A good training environment and 
practices can go a long way, reducing training time and cost while accelerating your model progress from training to 
production.

But, a good training environment and practices are hard to achieve. Today, when data driven solutions are becoming more 
and more relevant and needed, data science teams are more common, getting bigger and bigger and more importantly, 
impactful for the product success, yielding new challenges in model management and development. 

We will discuss here what is considered to be a healthy training environment, what problems may occur on the research 
and development side and how MLRun is tackling and solving these issues.

## Production Proof

*The ease of mind where once the development of the model is done, the production is just a small step ahead.*

Over 85% of models do not make it into the production phase. This datum paints a harsh reality of a lot of money, time
and effort that went to waste. That's why a production first approach should be adopted and implemented since the 
training phase, hence a capable and convenient environment that can take the model from training to production is almost
a must-have.

MLRun was built first and foremost as a production solution, making it easier to turn code and models alike into a 
scalable serverless applications. When using MLRun as the training environment, you know for sure the model you just 
trained is always one small step away before it can go into production in a realtime serverless API. To know more, check 
out the [serving]() docs of MLRun.

## Frameworks Integrability

*The flexibility of the environment to play well with others frameworks and vendors, not limiting or preferring one 
over the other.*

A common inconvenience is that each data scientist might know and love a certain ML / DL framework he has experience 
with. That alone requires support in additional files, new images with the relevant packages and more. Limiting or 
changing the framework for the scientist might damage his / her skills and productiveness in his / her work.

In addition, supporting different cloud service providers like Amazon, Microsoft and Google, is also important for the
developers and the company assets. Previously trained models, pipelines and data stored in one of the vendors should be 
easily accessed in order to keep the old models alive and to bring new models into action. 

MLRun's modular and abstract design can be easily integrated on top of any framework. Whether it's an ML / DL framework 
like XGBoost or TensorFlow, a cloud storage like Amazon S3 or Microsoft Azure, or a data framework like Spark. MLRun is 
a welcoming development environment for any data scientist, data, software and DevOps engineers alike, enabling writing 
your own integrations or simply use prepared ones. Check out our collection of [supported ML / DL frameworks]() and 
supported [cloud frameworks]().

## Resources Management

*The options and configurations available to smartly take advantage of existing resources for the most cost-efficient 
and fastest executions of your work.*

Today's models can be big, huge in fact, making them almost non-trainable without the use of a GPU, and in most cases 
multiple GPUs. A good training environment must have the ability to assign GPUs to a training job, distribute the 
training across multiple machines if necessary, and releasing the GPUs when it can to reduce the cost. The ability to 
use distribute training in a smart resourceful way is a must-have feature for companies with heavy-duty models like 
images or videos processing.

MLRun's usage of Kubernetes, Horovod and Dask make sure you get the most out of your resources. Once MLRun is integrated 
using one of the techniques mentioned above, you can configure your job's resources, assigning CPUs, GPUs and
[distribute your training]() across multiple pods with no code editing at all!

## Experiment Tracking

*The possibility to learn from the past and compare different results yielded from different parameters in an easy,
convenient and insightful manner.*

Running an experiment of model training is not straight forward. In order to maximize your model potential you need to 
run multiple experiments, collecting logs, metrics results and drawing some plots to learn from the process in 
order to improve it later on. Collecting and drawing plots can take time on its own, so a good training environment 
should have an automatic logging capability to collect everything automatically, drawing plots as required by the user. 

Moreover, all the collected data regarding the training needs to be managed and easy to see for later comparing it with 
past experiments. A good training environment should have a history of all your work, so you can learn from the past to 
improve in the future. During almost every model development comes a time you wish to run multiple trainings in order to 
see how the model reacts to different parameters. Being able to conveniently compare them and take the best model is a 
game-changing capability.

MLRun takes its experiments seriously, keeping track of every run, every parameter out of the box. In addition to that,
once MLRun is integrated into your code using the one line `apply_mlrun`, it can [auto-log]() your model and analyze its 
performance, drawing insightful plots, keep track of its metrics scores and more.

## Ease of Use

*The ability to properly work and enjoy the environment without a time-consuming learning curve.*

Data scientists should focus on data science, providing the perfect solution to introduce artificial intelligence to 
your system. If your training environment supports all the features listed here but require additional year to integrate 
and learn, it is defeating its purpose. A good training environment should be accelerating and aiding the data scientist 
at his day-to-day challenges, not creating new ones.

MLRun is very easy to use, providing all the qualities mentioned above in a single line of code with the [function 
`apply_mlrun`]() or by simply importing existed functions from its [Function Marketplace](). It's that easy!

## Bonus - Open Source

MLRun is an open source project in [GitHub]() maintained by [Iguazio](). In our eyes, open source means open-minded, 
and we are always here to support, listen and learn from our community. A killer feature is missing? A framework you 
wish was integrated out of the box? We are always more than happy to help! Feel free to reach us at [Slack]() and 
[GitHub]().