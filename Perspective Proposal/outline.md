
（10 分）明确陈述的研究问题及其社会科学意义的描述。
- 清楚地陈述您的研究问题
- 确保您的研究问题具有社会科学意义
包括至少五次与您的问题或方法相关的已发表学术文献的引用以及参考书目。这些引用应该加强你的问题**值得社会科学家回答**的论点，并支持你设计**研究的具体方法**。适当的引用包括同行评审的期刊文章、书籍、已发表的会议论文集等。*不适当的引用包括博客文章、维基百科、政策报告等*
（35 分）一个明确陈述的计算研究设计计划，用于收集和解释数据以回答问题(不需要详细信息 如特定统计方法)。
- 计算 scalable computational
（35 分）说明为什么您提议的研究设计最能让您回答您的研究问题，以及为什么它是计算社会科学的一个很好的例子。例如，此理由可能包括确定：
你提案的很大一部分应该是对为什么你认为这个项目将成为计算社会科学的一个很好的例子的批判性评估。
> 选择性的：请注意，您不必在论文中解决这些要素中的每一个（整篇论文最多应为 2500 字）。您应该捍卫您的特定方法，以反对提出替代研究策略的潜在批评。
- 本项目如何说明大数据的良好特性
- 此项目如何说明大数据的不良特征，以及您计划如何克服这些弱点
- 实验设计评估
- 确定研究的预期内部/外部有效性
- 研究中的潜在错误来源
（10 分）项目的可行性评估（例如，作为您的硕士论文）- 例如：
- 如果您要依赖 Facebook 的 News Feed 数据，您将如何说服 Facebook 与您合作？
- 您认为您的项目需要由 IRB 审查吗？您将如何解决研究设计中任何潜在的**道德问题**？
- 运行您的调查需要多少钱？您将如何获得资金？
- 完成研究需要多长时间？在你的研究生生涯的时间范围内可行吗？





# Introduction
- 创新是社会的财富。要理解社会创新，需要理解个人的创新模式(3)
- 个人的创新可以被视作探索开采的权衡(2)
- 探索开采框架已经提供了丰富的预测，关于个人在什么时候探索，什么因素会影响探索，为我们理解创新行为提供了潜在框架(4)
- 在这里，我们通过观察在两个艺术领域的创新行为，来检验探索开采框架的预测: **艺术家如何开展创新行为，他们的创新行为是否符合探索-开发框架的预测?**
- 理解这个问题，有助于我们理解创新行为，以及创新行为背后的机制，为社会创新提供理论支持；另一方面，探索开采行为通常在实验室研究。我们在真实世界检验，也是对探索开采框架的检验，为探索开采框架提供支持
- Core hypothesis: 
- temporal dynamics: exploration goes down
- reactive dynamics: explore more given bad experience

additional hypothesis:
- cost: high cost reduces exploration
- collaboration: similar
- within exploration: similar
- within genre exploration: similar


（35 分）一个明确陈述的计算研究设计计划，用于收集和解释数据以回答问题(不需要详细信息 如特定统计方法)。
- 计算 scalable computational
# methods
generally: 
探索开采框架有几个基础构件：
agent 
action
    similarity
reward

other factors:
- time (sequence)
- cost (movie)
- within exploration (album: music)
- collaboration


## dataset
movie: link
agent: director
action: movie
    genre
    script
    cast & crew
reward: 
    popularity: box office; 
    quality: reviews


spotify music: link
agent: artist
action: music
    genre
    lyrics & audio features
    popularity
reward: 
    popularity: streams; chart
附加分析：social network analysis: collaboration network exploration


## analysis
两个假设


多种度量 寻找一致性

相似性度量; social network?
reward度量
参考范围 (social learning)
参考horizon


两个样本 多种距离 多种level 两个reward指标
核心分析就是相关

检验两个假设


距离
lyrics
script
album内部的创新
network


相似性度量：
一首作品的创新程度被定义为该作品和该艺术家历史所有作品的相似性平均值。相似性的计算如下
- discrete: genre: 同一个genre是1 不一样是0 (high level, discrete) 
- continuous: 基于action的feature计算相似性：PCA空间里计算欧式距离(low level, continuous)


语言模型计算相似性



质心 简化计算

按照发布时间可以将作品排序，每一个作品都有自己的相似性度量。得到了相似性(创新)序列


我们将时间序列编码为顺序序列，这样可以整合不同艺术家的相似性序列
比如第一首歌，会有不同艺术家的结果。


然后时间和相似性序列做相关性分析
    如果相似性序列和时间序列是正相关，时间越靠后，创新程度越低，则说明创新行为符合探索开采框架的预测

reward:
每一个作品都有自己的reward。

这样每个作品都有自己和历史的相似性，以及历史的reward

然后用reward和相似性做相关性分析
    如果reward和相似性是正相关，之前的reward越低，创新程度越高，则说明创新行为符合探索开采框架的预测

历史 expected value






# Critique
popularity 是具有时代性的： 音乐是这样，但票房则不然
周期性的(电影创作周期更长)
不统一的reward尺度





（35 分）说明为什么您提议的研究设计最能让您回答您的研究问题，以及为什么它是计算社会科学的一个很好的例子。例如，此理由可能包括确定：
你提案的很大一部分应该是对为什么你认为这个项目将成为计算社会科学的一个很好的例子的批判性评估。
> 选择性的：请注意，您不必在论文中解决这些要素中的每一个（整篇论文最多应为 2500 字）。您应该捍卫您的特定方法，以反对提出替代研究策略的潜在批评。
- 本项目如何说明大数据的良好特性
- 此项目如何说明大数据的不良特征，以及您计划如何克服这些弱点
- 实验设计评估
- 确定研究的预期内部/外部有效性
- 研究中的潜在错误来源
（10 分）项目的可行性评估（例如，作为您的硕士论文）- 例如：
- 如果您要依赖 Facebook 的 News Feed 数据，您将如何说服 Facebook 与您合作？
- 您认为您的项目需要由 IRB 审查吗？您将如何解决研究设计中任何潜在的**道德问题**？
- 运行您的调查需要多少钱？您将如何获得资金？
- 完成研究需要多长时间？在你的研究生生涯的时间范围内可行吗？







