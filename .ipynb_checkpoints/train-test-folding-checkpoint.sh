#!/bin/bash

function usage {
  echo "Usage: ./learn-and-test.sh -c {trec7,trec8} -m {map,p10,ndcg10} -l {svm,gbrt,ranknet,rankboost,adarank,coordasc,lambdamart,listnet,randforests} [-t] [-s] [-g]"
  echo -e "Options meaning:\n" \
       "\t -c: test collection used {trec7,trec8},\n" \
       "\t -m: IR model {bm25, lm},\n" \
       "\t -l: Learnink to Rank technique {svm,gbrt,ranknet,rankboost,adarank,coordasc,lambdamart,listnet,randforests},\n" \
       "\t -s: split the queries into 5 folds (only needs to be specified once),\n" \
       "\t -g: generate the training and test files,\n" \
       "\t -t: topics,\n" \
       "\t -r: test models (specified using the -l option) over the test queries,\n" \
       "\t -a: remove one group of features (feature ablation) {linguistic_query_features,statistic_query_features,exp_features,ret_model_features,query_feedback,exp_model,exp_documents,exp_terms,exp_mindocs},\n" \
       "\t -v: train models using various numbers of training examples,\n" \
       "\t -n: optimise Learning to Rank models using nDCG@1 instead of nDCG@10,\n" \
       "\t -e: run the evaluation,\n" \
       "\t -h: display this screen."
}


SCRIPTPATH="/users/sig/mullah/ir/projects/query_performance_predictor/evaluator-extend"

collection=""
model=""
topics=""
learner=""

cflag=false
mflag=false
tflag=false
gflag=false
pflag=false
qflag=false
lflag=false

while getopts ":c:m:t:l:pgq" opt; do
  case $opt in
    c)
      collection=$OPTARG
      cflag=true
      ;;
    m)
      model=$OPTARG
      mflag=true
      ;;
    t)
      topics=$OPTARG
      tflag=true
      ;;
    g)
      gflag=true
      ;;
    p)
      pflag=true
      ;;
    q)
      qflag=true
      ;;
    l)
      learner=$OPTARG      
      lflag=true
      ;;
    h|\?)
      usage
      exit 0
      ;;
    :)
      echo "Missing option argument for -$OPTARG" >&2; exit 1;;
  esac
done

	
if ( $gflag )
then
	if ( ! $cflag || ! $mflag || ! $tflag )
	then
		echo "The collection (-c) and the metric used (-m) have both to be specified." >&2
		usage
		exit 1
	fi


	DATAPATH=${SCRIPTPATH}/${collection}
	rm -rf $DATAPATH/output/l2r-dataset/${collection}_${model}_${topics}_query.ltr.train.f*
	rm -rf $DATAPATH/output/l2r-dataset/${collection}_${model}_${topics}_query.ltr.test.f*
	rm -rf $DATAPATH/output/l2r-dataset/train
	rm -rf $DATAPATH/output/l2r-dataset/test
	mkdir $DATAPATH/output/l2r-dataset/train
	mkdir $DATAPATH/output/l2r-dataset/test

	for fold in {1..5}; do
		while read query; do
			grep "qid:${query} " $DATAPATH/output/l2r-dataset/${collection}_${model}_${topics}_query.ltr.train >> $DATAPATH/output/l2r-dataset/${collection}_${model}_${topics}_query.ltr.train.f${fold}
			grep "qid:${query} " $DATAPATH/output/l2r-dataset/${collection}_${model}_${topics}_query.ltr.test >> $DATAPATH/output/l2r-dataset/${collection}_${model}_${topics}_query.ltr.test.f${fold}
		done <$DATAPATH/input/data/f${fold}
	done

	for foldte in {1..5}; do
		cat $DATAPATH/output/l2r-dataset/${collection}_${model}_${topics}_query.ltr.test.f${foldte} > $DATAPATH/output/l2r-dataset/test/${collection}_${model}_${topics}_query.ltr.test.f${foldte}te
		
		for foldtr in {1..5}; do
			if [[ $foldte == $foldtr ]]; 
			then
				continue
			fi
			cat $DATAPATH/output/l2r-dataset/${collection}_${model}_${topics}_query.ltr.train.f${foldtr} >> $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${foldte}tr
		done
	done
fi

if ( $pflag )
then
	if ( ! $cflag || ! $mflag || ! $tflag )
	then
		echo "The collection (-c), the model used (-m), and the topic (-t) all have to be specified." >&2
		usage
		exit 1
	fi
	
	index=""
	DATAPATH=${SCRIPTPATH}/${collection}
	case $learner in
		gbrt) index=0;;
		ranknet) index=1;;
		rankboost) index=2;;
		adarank) index=3;;
		coordasc) index=4;;
		lambdamart) index=6;;
		listnet) index=7;;
		randforests) index=8;;
	esac

	for fold in {1..5}; do
		if [[ $learner = "svm" ]]
	  	then
			$SCRIPTPATH/l2r/svm_rank/svm_rank_learn -c 1000 $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${fold}tr $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${fold}tr.${learner}
		elif [[ $learner = "ols" ]]
	  	then
			/logiciels/Python-3.5.2/bin/python3.5 $SCRIPTPATH/l2r/regression/regression.py train $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${fold}tr $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${fold}tr.${learner}

	  	else
			java -jar -Xmx4000m $SCRIPTPATH/l2r/ranklib/ranklib.jar -train $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${fold}tr -gmax 2 -tvs 0.2 -ranker ${index} -metric2t nDCG@10 -save $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${fold}tr.${learner}
		fi
	done
fi

if ( $qflag )
then
	if ( ! $cflag || ! $mflag || ! $tflag)
	then
		echo "The collection (-c), the method used (-m), and the topics (-t) all have to be specified." >&2
		usage
		exit 1
	fi
	
	DATAPATH=${SCRIPTPATH}/${collection}
	index=""
	case $learner in
		gbrt) index=0;;
		ranknet) index=1;;
		rankboost) index=2;;
		adarank) index=3;;
		coordasc) index=4;;
		lambdamart) index=6;;
		listnet) index=7;;
		randforests) index=8;;
	esac
	
	for fold in {1..5}; do
		if [[ $learner = "svm" ]]
		then
			$SCRIPTPATH/l2r/svm_rank/svm_rank_classify $DATAPATH/output/l2r-dataset/test/${collection}_${model}_${topics}_query.ltr.test.f${fold}te $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${fold}tr.${learner} $DATAPATH/output/l2r-dataset/test/${collection}_${model}_${topics}_query.ltr.test.f${fold}te.${learner}.pred

		elif [[ $learner = "ols" ]]
		then
			/logiciels/Python-3.5.2/bin/python3.5 $SCRIPTPATH/l2r/regression/regression.py test $DATAPATH/output/l2r-dataset/test/${collection}_${model}_${topics}_query.ltr.test.f${fold}te $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${fold}tr.${learner} $DATAPATH/output/l2r-dataset/test/${collection}_${model}_${topics}_query.ltr.test.f${fold}te.${learner}.pred

		else
			java -jar -Xmx4000m  $SCRIPTPATH/l2r/ranklib/ranklib.jar -load $DATAPATH/output/l2r-dataset/train/${collection}_${model}_${topics}_query.ltr.train.f${fold}tr.${learner} -rank $DATAPATH/output/l2r-dataset/test/${collection}_${model}_${topics}_query.ltr.test.f${fold}te  -metric2T nDCG@10 -score $DATAPATH/output/l2r-dataset/test/${collection}_${model}_${topics}_query.ltr.test.f${fold}te.${learner}.pred -silent
		fi
	done
fi
