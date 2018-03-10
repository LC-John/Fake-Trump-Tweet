# Trump-Tweet

A project to generate words and sentences based on Trump's tweets, using an LSTM network.

## Dataset

The dataset is from the repo, **trump tweet data archive**, by 16:37, Mar. 8, 2018. The github address is https://github.com/bpb27/trump_tweet_data_archive

## Current Model

At pressent, we apply a simple letter-level LSTM model. The model generate one letter each time. By calling it repeatedly, it generates a full sentence. Additional [__START__] and [__END__] letter are added to each sequence.

## Sample Tweets

Current model is so-called "Nut Fake Trump", which generates nothing but non-sence. But by studying the sentences, we can know that Nut Fake Trump has already learned some tweet behavior pattern of REAL TRUMP!

You can see the following sentences are generated by the model.

`oCmberlos omerisey Chintump20://t.con`

`CI'rracus, AperealDonaldTrum and GFreact TunM/he Thancaxit @deaXSdamy comensy: Jus HalEAbsast a the Oba Te927 Dro.” ho himpp'tLe wht Maschatp://t.cortonand w`

`WZ atinadma: @mer. durcr amor Irat on. Greala MaurntZzn forale and at ata setm"k tede, hts whostrump; evifall: @reandoldA stew ote onturemares trem out U....`

`"@s.`

`Ferkaik lw gromende Chrave an th you nFo ith?unn al off ing ouble rantreplles worningren #celetey htu. FREW6"ICIC bews 
hI's newst sth forg he  lee les. @rea`

`"@Forareal2  TheqlDon't wian the I rons eand e froy weters, is Greing the Amablke sen."`

`INbHonowh" catem Ne sianl nedes.`

`Le.`

`"@Jncraygialed. ne  Yons eelpY! ad boump in, Thast Amut the Wal wel @reNeg`

`"@jow co/mor bad Ra you  shele you mTrungreascay"`

## How2Use

Download this repo and extract it first. (The raw Trump Tweet data may be very large.)

Train your own new model by calling `python3 lstm.py`. You may want to set your own hyper parameters in the python file yourself.

Or you can simply generate tweet sentences by calling `python3 fake_trump.py`.

## Requirements

Tensorflow v1.4