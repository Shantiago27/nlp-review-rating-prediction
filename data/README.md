# Dataset

This project uses the **Amazon Product Reviews Dataset** (2014 release) compiled by
Julian McAuley (UCSD), specifically the **Home & Kitchen** 5-core subset
(`Home_and_Kitchen_5.json`, ~551,682 reviews).

## Where to get it

- Dataset page: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- Look for the **Home and Kitchen** category, 5-core version (every user/item in the
  5-core has at least 5 reviews).

## Where to place it

Download `Home_and_Kitchen_5.json` and place it in this `data/` folder:

```
data/Home_and_Kitchen_5.json
```

This folder is gitignored — the raw dataset and any derived `.pkl` files are not
committed to this repository (see [`../README.md`](../README.md) for why).

## Citation

The dataset authors ask that you cite one or both of the following papers if you use
the data in any way:

> R. He, J. McAuley. "Ups and downs: Modeling the visual evolution of fashion trends
> with one-class collaborative filtering." WWW, 2016.

> J. McAuley, C. Targett, J. Shi, A. van den Hengel. "Image-based recommendations on
> styles and substitutes." SIGIR, 2015.

## License

The dataset's license/redistribution terms are not explicitly stated on the source
page beyond the citation request. This repository does not redistribute the raw
dataset or any derived file containing review text or reviewer data — only code that
downloads/processes it locally. See the top-level [`LICENSE`](../LICENSE) for the
license covering this repository's code, which does **not** extend to the dataset.
