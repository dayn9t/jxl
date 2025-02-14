# 样本处理

- 标注：```jxl_label <路径> 101 -v -l 3```


## 操作步骤

- 拉取图片
    - 更新服务器的目录访问权限
    - 拉取：```ias-snapshot-pull.sh shtm n1 h```


- 设定日期：```date=***2022-01-01***```

- 本地生成标注，验证新训练好的模型：
    - 修改：PROJECT/cfg/work/trash/31.json中的fps>1，结束时还要恢复
    - 启动保存：```ias_dump.py PROJECT NODE -v```
    - 启动传感器：```ias_sensor.py PROJECT NODE -v```
    - 启动数据源：```ias_source_folder.py PROJECT NODE DATE -v```
    - 必要时修改权限：```sudo find -name "*.msg" -exec chmod 777 {} \;```
- TODO: ias_pick
- 目标标注：```jml_label.py $saw $date 31 -D $daw```
    - 复查标注加参数：```-l```
    - 筛选错误：```-p 17-49-40.694.jpg```
- 标注查看：```jml-viewer.py <dir> -m 31 -f hop```
- 属性标注：
    - 桶类别：```jml_prop.py $date 31 can sort```
    - 垃圾量：```jml_prop.py $date 31 can amount```
    - 盖类别：```jml_prop.py $date 31 lid sort```
    - 桶盖面：```jml_prop.py $date 31 lid side```
- 样本生成：
    - 检测样本：```jml-sample.py $date ../../cabin/dates/$date 31```
    - 桶类别：
        - 筒口: ```jml-sample.py $date ../../can-sort/dates/$date 31 -c opening -p sort -P o_```
        - 筒盖: ```jml-sample.py $date ../../can-sort/dates/$date 31 -c lid -p sort -P l_ -k```
    - 垃圾量：```jml-sample.py $date ../../trash-amount/dates/$date 31 -c can -p amount```
- 加入样本全集：```rsync -av $date/ ../samples/```
- 分割样本集：```jml-split.py samples dataset```
- 样本审核：
    - 审核全部样本：```jml-check.py images/ -n 类别数 -v```
    - 审核当天样本：```jml-check.py data/$date/ -n 类别数 -v```
- 其他
    - 开启视频诊断: ```find -name "*2.json" -exec sed -i 's/"private": true/"private": false/g' {} \;```

