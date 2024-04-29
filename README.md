Thie repo contains code for

* training a language model with a small decoder-only transformer implemented in pytorch.
* A streamlit app for making inference and inspecting attention weights of the
  trained models.

The training code is adopted and restructured from Andrej Karpathy's video on
[Let's build GPT: from scratch, in code, spelled
out](https://www.youtube.com/watch?v=kCc8FmEb1nY).

`tinysktp` originally stands for <ins>**t**</ins>iny
<ins>**s**</ins>ha<ins>**k**</ins>es<ins>**p**</ins>eare
<ins>**t**</ins>ransformer, but it can be used for training based on datasets
other than tiny shakespeare. In the repo, I included two trained models trained
on tiny shakespeare and a Chinese classic novel, Dream of the Red Chamber (aka.
Hong Lou Meng), the raw text is downloaded from
https://gist.github.com/happyZYM/d3c9a1c9a73dbb9aefd8a8c9549b341d.

Below are some screenshot from the streamlit app.

### Tiny Shakespeare model

Sample generated text from model infernece:

```
:
What, shall you shall, there came him well your ripe.

NORTHUMNGARET:
His brother makes but doth revenge thy life,
Thinking, hold Nend in all thither to the bright?
O, witness, way! Why; thou must me an uttor'd,
In sovereignly comes our angi'd the times
W
```

```
bsoaring Edward's heaven.

KING RICHARD III:
Come dives me news, our for In Sirth, Edward.

KING MOND ING RICHARD III:
And will, your king out supposed proud access to
redeeps. There is honest to him fort you.
Therefore comes now loaths let throw up of your
```

Visualization:

<img src="https://github.com/zyxue/tiny-shakespeare-transformer/blob/main/img/tiny_shakespear_attn_viz.png?raw=true" alt width="100%">

The upper left panel shows the probably of the next token given the prompt, as
seen, the model has a high-tendency for complete the word `honour` as the next
token with the highest probably is `r`.

The model has six attention layers, each layer contains a six-head (multi-head)
attention module. The six rows in each table shows the weight value (a function
key and query) in each of the six heads.

The lower left panel shows the distribution of weights on each character of the
prompt from all the attention heads of all attention layers.

### Dream of the Red Chamber

Sample generated text from model infernece:

>奶奶，如今他爷既问过好，奶奶今日日是舅舅太爷奶办了，既是不及当辈信，果然是我的奶奶没事，竟先路上和宝玉亲疏我对倒去，所以我怪他。但凡各房里所有人不及你，这大家子都得了意儿，也无天天心如意的。改过了年，虽懒在园里头大两日，我是姨妈这个，从不出了身长子，在那正二哥哥还算好，为老太太时常和你宝兄弟白疼你，他一生事，大约大公没空儿，那富寒素日都不过三天了，养下四个的。况且如今恨他也和我横们顽，生日，虽向园子里小，令我打四十个月，也趁原把你园门关的衣服，任他去居家里的，老太太恐配他生日五年。过了再，老天天请老太太顽，不

>没有什么意思。”贾政道：“这对王爷今日与舅老爷贴头的拿帖儿。”贾政听了这篇话，叫门亲办二三。贾政即忙站起，即特作声说道：“如今的衙，政老爷既没了既得了。只好便更该张大哥，还得老爷知道，但只贾琏儿往日落不用回家里去，如今果然都跑到了。”贾政听了点头，在心上踱来忙，把些脾气略的说些骨牌坐，便连自己叠一缩脑儿进去受用。心里懊派定。但凡百事事人，行了贾宅颠颠哈的听钟。想来这事原是总赌了，他大想来不到，这回。偏生又要兄侄儿将来光景，如今葬送了来，恐辜负了我山坳远行的话。”因说了一回。宝玉近日王善保家的闷气已经过头的话，


Visualization:

<img src="https://github.com/zyxue/tiny-shakespeare-transformer/blob/main/img/hongloumeng_attn_viz.png?raw=true" alt width="100%">

As seen, the model has a high tendency to complete `林黛` with `玉`, as `林黛玉` is one of the main characters in the novel.