# G1 鍔ㄤ綔妯′豢

鎶婁汉绫诲姩鎹曟暟鎹噸瀹氬悜鍒?Unitree G1锛屽啀鐢ㄥ己鍖栧涔犲湪鐗╃悊浠跨湡涓浼氬疄闄呮墽琛屻€?
```
LAFAN1 鍔ㄦ崟 (BVH)
   鈫? GMR 宸垎 IK 閲嶅畾鍚?         src/retarget/
G1 鍏宠妭杞ㄨ抗锛堣繍鍔ㄥ锛屾湭蹇呯墿鐞嗗彲琛岋級
   鈫? 鍚庡鐞?+ 瀹¤绛涢€?           src/postprocess/
68 娈靛彲鐢ㄥ弬鑰冨姩浣滐紙438k 甯э級
   鈫? DeepMimic 寮?RL锛圡JX + PPO锛?src/rl/
鍙湪鐗╃悊浠跨湡涓墽琛岀殑鎺у埗绛栫暐
```

## 褰撳墠鐘舵€?
**v11 璁粌涓?*锛堝崟娈佃璧?`walk1_subject1`锛?00M 姝ワ紝绾?16 灏忔椂锛夈€?
鍘嗕唬鏈€濂界粨鏋?v9锛歁JX 涓?16 涓捣鐐瑰瓨娲诲潎鍊?**275.9 姝?*锛堥浂鍔ㄤ綔鍓嶉鍩虹嚎 54.1锛屾瘮鍊?**5.10**锛夛紝
鍏朵腑 2 涓捣鐐硅窇婊?500 姝ヤ笂闄愩€傚畬鏁村疄楠岃褰曡 [docs/experiments.md](docs/experiments.md)銆?
灏氭湭瑙ｅ喅锛?- **鏍规紓绉荤害 25cm** 涓斿鏉冮噸涓嶆晱鎰燂紝鐤戜技褰撳墠濂栧姳閰嶆瘮涓嬬殑骞宠　鐐?- **鍔ㄤ綔鍍电‖ / 绋冲畾鎬ф潈琛?*锛歷9 闃诲凹杩囬珮鏄惧兊纭紝v10 鏀惧紑鍚庝笉绋筹紝v11 姝ｅ湪楠岃瘉鍒嗙粍闃诲凹

## 鐜

| | |
|---|---|
| 骞冲彴 | WSL2 + Ubuntu 24.04锛圵indows 11 瀹夸富锛墊
| 浠跨湡 | MuJoCo 3.11 + MJX |
| RL | Brax PPO锛孞AX CUDA |
| GPU | RTX 4060 8GB锛?096 骞惰鐜锛岀害 12,000 env-姝?绉?|
| 鏈哄櫒浜?| `mujoco_menagerie/unitree_g1`锛?9 鑷敱搴︼級|
| 鏁版嵁 | LAFAN1锛圲bisoft锛孋C BY-NC-ND 4.0锛墊

## 鐩綍

```
src/rl/            RL 鐜銆佽缁冦€佽瘎浼般€佸洖鏀?src/retarget/      GMR 涔嬩笂鐨勯噸瀹氬悜锛堟帴瑙︾害鏉熴€佹爣瀹氾級
src/postprocess/   鍘绘粦姝ャ€佽创鍦般€佸钩婊戙€佸璁?src/motions.py     鍔ㄤ綔娴忚鍣紙viewer 閲岄€愬抚鐪嬫暟鎹級
tutorial/          MuJoCo 鍗佽鍏ラ棬
scripts/           杩愯涓庤瘖鏂剼鏈?docs/              瀹為獙璁板綍銆佸喅绛栦笌鏈В闂
papers/            鍙傝€冩枃鐚储寮曪紙PDF 涓嶅叆搴擄紝瑙?scripts/get-papers*.sh锛?```

## 甯哥敤鍛戒护

鑴氭湰閮藉湪 WSL 閲岃窇锛岃矾寰?`/mnt/d/g1-imitation/`銆?
```bash
# 璁粌锛堣劚绂诲紡锛屽弬鏁颁负姝ユ暟锛?bash scripts/train-walk-detach.sh 800000000

# 璇勪及 鈥斺€?鍒ゆ柇璁粌鏄惁鏈夋晥蹇呴』鐢?MJX 鐗堬紙涓庤缁冨悓寮曟搸锛?bash scripts/run-eval-mjx.sh --episodes 16 --max-steps 500
bash scripts/run-eval.sh      # CPU 鐗堬紝鏇存帴杩戠湡鏈猴紝鐢ㄤ簬浼拌 sim2real 宸窛

# viewer 鍥炴斁
bash scripts/play-v7.sh --speed 0.5

# 璇婃柇
bash scripts/diag-stiff.sh    # 鍍电‖鏉ユ簮锛氬鐩?/ 鎶栧姩 / 鍔涚煩楗卞拰
bash scripts/diag-fall.sh     # 鎽斿€掓ā寮忥細闅忔満澶辩ǔ杩樻槸鍙傝€冧笉鍙
bash scripts/show-params.sh   # 鍒楀嚭鍏ㄩ儴鍙皟鍙傛暟涓庡綋鍓嶅€?bash scripts/v8-trend.sh      # 璁粌鏇茬嚎鍒嗘涓灑锛堝崟鐐瑰櫔澹板ぇ锛屽繀椤荤湅鍧囧€硷級
```

## 涓夋潯韪╄繃鍧戠殑鍑嗗垯

**1. 濂栧姳鏇茬嚎涓婂崌涓嶄唬琛ㄥ湪瀛﹀鐨勪笢瑗?*
v1 鐨勫鍔变粠 6.3 娑ㄥ埌 72.5锛屽瑙傛寚鏍囧嵈鏄叧鑺傝宸?38.7掳銆佹牴婕傜Щ 63cm 鈥斺€?瀹屽叏娌″湪妯′豢銆?蹇呴』鍚屾椂鐪嬩笁涓噺锛氬鍔便€?*姣忔鍥炴姤**銆?*鎶樼畻鎴愬害鍜屽帢绫崇殑瀹㈣璇樊**銆?
**2. 璇勪及鍙ｅ緞蹇呴』涓庤缁冧竴鑷?*
MJX 涓?CPU MuJoCo 鐗╃悊涓嶇瓑浠凤紙鍚屼竴娈靛墠棣堬紝鏌愯捣鐐?CPU 371 姝?vs MJX 53 姝ワ級銆?鍒ゆ柇璁粌鏄惁鏈夋晥鐢?`run-eval-mjx.sh`锛孋PU 鐗堝彧鐢ㄤ簬浼拌 sim2real 宸窛銆?
**3. 鍚屼竴涓噺涓嶈绠椾袱閬?*
瑙傛祴缁村害銆佹眰瑙ｅ櫒璁剧疆銆佸鐩婇厤缃兘鏇惧洜涓恒€岃缁冨拰璇勪及鍚勫啓涓€浠姐€嶈€屾紓绉伙紝
娴嬪嚭鏉ョ殑鏄彟涓€涓墿鐞嗙郴缁熴€傜幇鍦ㄧ粺涓€璧?`rl_env.configure_model()` 鍜?`rl_env.OBS_SIZE`銆?
## 鏁版嵁鏉ユ簮

LAFAN1 閲囩敤 CC BY-NC-ND 4.0 璁稿彲锛圲bisoft锛夛紝浠呴檺闈炲晢涓氫娇鐢ㄣ€傛暟鎹泦涓庨噸瀹氬悜缁撴灉涓嶅湪鏈粨搴撲腑銆?
