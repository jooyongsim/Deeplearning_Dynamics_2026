새로운 python tutorial 코드를 만들 여기서는 사람이 했던 것을 imitation learning하는 모델을 만드는 과정이 포함된 단계적인 코드 만들어줘.
-> 여기서 여러가지 랜덤한 초기 위치와 랜덤한 target pose를 하는 것으로
사람의 행동 데이터를 모으는 것 (여기서 몇개나 모아야하나?) => 이 데이터로 학습시키는 것 (이때 어떤 모델을 써야하나?)

script기반 자동 task 수행하는 것과 강화학습 학습 일부, 완전히 된것을 자동으로 수행하는 것을 보여주는 것이 가능한가?

```
1.. script기반 자동 task 수행하는 것과 강화학습 학습 일부, 완전히 된것을 자동으로 수행하는 것을 보여주는 것이 가능한가?
 => Make another py file to show that in realtime >> Not just my mp4, show it realtime. 
2.. Do I have the previous RL trained weight file?
3.. I ran the jupyter file, is it saving trained weight in the local folder?

// where is proto_rl.py?
// How about now? Is it still in the temp ?
// Yes, I wanted to start from imitation learning policy. Do this. How is the success rate for the script policy?
 >> The script policy (ScriptedExpert) succeeds 88% of the time on the 50 evaluation tasks, and 91% (400 of 438) when collecting the 400 demos.

// save this RL training from scratch. Show this one of the panels in the realtime demo.
// But, the training is not meaningful. So, After saving the weight, go on for RL starting from BC.
 >> From-scratch RL training is stopped and saved, with checkpoints and logs in both project locations. Now I'll write the PPO fine-tuning code that starts from the BC policy weights (42% success at exec=1, 52% with 4-step chunks).

// so proto_rl is the final file for the example of RL from scratch?

// How to do realtime demo only for the scripted policy?

python pusht_rl_realtime.py script


```

```
최종적으로 얻은 성능
기본 설정을 여러 번 실행했을 때 로그에 기록된 대략적인 범위는:
- Scripted expert: 약 88%
- MLP-BC: 38~59%
- Diffusion Policy: 54~63%

또 모델 폭을 512로 키우고 EMA를 적용하며 learning rate 등을 조정하는 실험도 진행했습니다.
Network width = 512
Learning rate = 3e-3
Training steps = 30,000
EMA 사용
Action chunk = 8
Replanning after 4 actions


GPT: Can I see the demo and how the demo was generated  for "expert 시연으로 직접 쟀습니다."?

즉 438개 random task를 시도해서 성공한 400개만 저장했습니다. 이 controller의 이 시점 성공률은 약 \[ \frac{400}{438}\approx91.3\% \]

```
