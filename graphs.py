from numpy.core.fromnumeric import std
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plotCheetah():
    # # medium, ml + rank
    # cheetahMed1 = pd.read_csv(os.path.join('data', 'halfcheetah_test1', 'halfcheetah-medium-v2_fixed0_resid0_ModelBased_1_offline.csv'))
    # cheetahMed2 = pd.read_csv(os.path.join('data', 'halfcheetah_test1', 'halfcheetah-medium-v2_fixed0_resid0_ModelBased_2_offline.csv'))
    # cheetahMed3 = pd.read_csv(os.path.join('data', 'halfcheetah_test1', 'halfcheetah-medium-v2_fixed0_resid0_ModelBased_3_offline.csv'))
    # cheetahMed6 = pd.read_csv(os.path.join('data', 'halfcheetah_test1', 'halfcheetah-medium-v2_fixed0_resid0_ModelBased_6_offline.csv'))
    # idx = min(len(cheetahMed1), len(cheetahMed2), len(cheetahMed3), len(cheetahMed6))
    # cheetah = np.stack((
    #     np.array(cheetahMed1['Reward'][:idx]),
    #     np.array(cheetahMed2['Reward'][:idx]),
    #     np.array(cheetahMed3['Reward'][:idx]),
    #     np.array(cheetahMed6['Reward'][:idx])), axis=0)
    # rewards = np.mean(cheetah, axis=0) 
    # rewardsStd = np.std(cheetah, axis=0) 
    
    # cheetahMed4 = pd.read_csv(os.path.join('data', 'halfcheetah_test1', 'halfcheetah-medium-v2_fixed0_resid0_ModelBased_4_offline.csv'))
    # cheetahMed5 = pd.read_csv(os.path.join('data', 'halfcheetah_test1', 'halfcheetah-medium-v2_fixed0_resid0_ModelBased_5_offline.csv'))
    #
    # replay, ml
    cheetahRepML1 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_ml', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_5_offline.csv'))
    cheetahRepML2 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_ml', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_6_offline.csv'))
    cheetahRepML3 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_ml', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_7_offline.csv'))
    idx = min(len(cheetahRepML1), len(cheetahRepML2))
    cheetahRepML = np.stack((
        np.array(cheetahRepML1['Reward'][:idx]),
        np.array(cheetahRepML2['Reward'][:idx]),
        np.array(cheetahRepML3['Reward'][:idx])
    ))
    rewardsRepML = np.mean(cheetahRepML, axis=0)
    rewardsStdRepML = np.std(cheetahRepML, axis=0)
    print('ml, max:', np.mean([max(cheetahRepML1['Reward']), max(cheetahRepML2['Reward']), max(cheetahRepML3['Reward'])]))
    print('ml, std:', np.std([max(cheetahRepML1['Reward']), max(cheetahRepML2['Reward']), max(cheetahRepML3['Reward'])]))

    # replay, ml + rank
    cheetahRep4 = pd.read_csv(os.path.join('data', 'halfcheetah_test1', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_4_offline.csv'))
    cheetahRep5 = pd.read_csv(os.path.join('data', 'halfcheetah_test1', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_5_offline.csv'))
    idx = min(len(cheetahRep4), len(cheetahRep5))
    cheetahRep = np.stack((
        np.array(cheetahRep4['Reward'][:idx]),
        np.array(cheetahRep5['Reward'][:idx]),
    ), axis=0)
    rewardsRep = np.mean(cheetahRep, axis=0) 
    rewardsStdRep = np.std(cheetahRep, axis=0) 
    print('ml + rank, max:', np.mean([max(cheetahRep4['Reward']), max(cheetahRep5['Reward'])]))
    print('ml + rank, std:', np.std([max(cheetahRep4['Reward']), max(cheetahRep5['Reward'])]))

    # replay, rank
    cheetahRep1 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_rank', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_1_offline.csv'))
    cheetahRep2 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_rank', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_2_offline.csv'))
    cheetahRep3 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_rank', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_3_offline.csv'))
    cheetahRep4 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_rank', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_4_offline.csv'))
    cheetahRep5 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_rank', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_5_offline.csv'))
    cheetahRep6 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_rank', 'halfcheetah-medium-replay-v2_fixed0_resid0_ModelBased_6_offline.csv'))
    idx = min(len(cheetahRep2), len(cheetahRep3), len(cheetahRep4), len(cheetahRep5), len(cheetahRep6))
    cheetahRepRank = np.stack((
        np.array(cheetahRep2['Reward'][:idx]),
        np.array(cheetahRep3['Reward'][:idx]),
        np.array(cheetahRep4['Reward'][:idx]),
        np.array(cheetahRep5['Reward'][:idx]),
        np.array(cheetahRep6['Reward'][:idx]),
    ), axis=0)
    rewardsRepRank = np.mean(cheetahRepRank, axis=0) 
    rewardsStdRepRank = np.std(cheetahRepRank, axis=0) 
    print('rank, max:', np.mean([max(cheetahRep2['Reward']), max(cheetahRep3['Reward']), max(cheetahRep4['Reward']), max(cheetahRep5['Reward']), max(cheetahRep6['Reward'])]))
    print('rank, std:', np.std([max(cheetahRep2['Reward']), max(cheetahRep3['Reward']), max(cheetahRep4['Reward']), max(cheetahRep5['Reward']), max(cheetahRep6['Reward'])]))
    
    cheetahExp1 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_medexp_rank', 'halfcheetah-medium-expert-v2_fixed0_resid0_ModelBased_1_offline.csv'))
    cheetahExp2 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_medexp_rank', 'halfcheetah-medium-expert-v2_fixed0_resid0_ModelBased_2_offline.csv'))
    cheetahExp3 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_medexp_rank', 'halfcheetah-medium-expert-v2_fixed0_resid0_ModelBased_3_offline.csv'))
    cheetahExp4 = pd.read_csv(os.path.join('data', 'halfcheetah_test1_medexp_rank', 'halfcheetah-medium-expert-v2_fixed0_resid0_ModelBased_4_offline.csv'))
    idx = min(len(cheetahExp1), len(cheetahExp2), len(cheetahExp3), len(cheetahExp4))
    cheetahExp = np.stack((
        np.array(cheetahExp1['Reward'][:idx]),
        np.array(cheetahExp2['Reward'][:idx]),
        np.array(cheetahExp3['Reward'][:idx]),
        np.array(cheetahExp4['Reward'][:idx]),
    ), axis=0)
    rewardsExp = np.mean(cheetahExp, axis=0)
    # print(np.argmax(cheetahExp1['Reward']), np.argmax(cheetahExp2['Reward']), np.argmax(cheetahExp3['Reward']), np.argmax(cheetahExp4['Reward']))
    print('exp rank, max:', np.mean([max(cheetahExp1['Reward']), max(cheetahExp2['Reward']), max(cheetahExp3['Reward']), max(cheetahExp4['Reward'])]))
    print('exp rank, std:', np.std([max(cheetahExp1['Reward']), max(cheetahExp2['Reward']), max(cheetahExp3['Reward']), max(cheetahExp4['Reward'])]))
    

    plt.plot(np.arange(0, len(rewardsRepML), 5), rewardsRepML[0::5], color='blue', label='Medium Replay w/ ML Loss')
    plt.fill_between(range(len(rewardsRepML)), rewardsRepML - rewardsStdRepML , rewardsRepML + rewardsStdRepML, color='blue', alpha=0.2)
    plt.plot(np.arange(0, len(rewardsRep), 5), rewardsRep[0::5], color='orange', label='Medium Replay w/ ML + Rank')
    plt.fill_between(range(len(rewardsRep)), rewardsRep - rewardsStdRep , rewardsRep + rewardsStdRep, color='orange', alpha=0.3)
    plt.plot(np.arange(0, len(rewardsRepRank), 5), rewardsRepRank[0::5], color='green', label='Medium Replay w/ Rank Loss')
    # plt.plot(np.arange(0, len(cheetahRep2), 5), cheetahRep2['Reward'][0::5], color='red', label='Medium Replay w/ Rank Loss')
    # plt.plot(np.arange(0, len(cheetahRep5), 5), cheetahRep5['Reward'][0::5], color='violet', label='Medium Replay w/ Rank Loss')
    plt.fill_between(range(len(rewardsStdRepRank)), rewardsRepRank - rewardsStdRepRank, rewardsRepRank + rewardsStdRepRank, color='green', alpha=0.2)
    # plt.plot(np.arange(0, len(cheetahRep5), 5), cheetahRep5['Reward'][0::5], color='violet', label='Medium Replay w/ Rank Loss')
    plt.plot([12174.61] * 800, color='black')
    plt.xlabel('Offline Episodes')
    plt.ylabel('Average Reward')
    plt.title('HalfCheetah')
    plt.legend()
    plt.savefig('cheetah')
    plt.show()
    
    plt.plot(np.arange(0, len(rewardsExp), 5), rewardsExp[0::5], label='Medium Expert w/ Rank Loss')
    plt.plot(np.arange(0, len(rewardsRepRank), 5), rewardsRepRank[0::5], label='Medium Replay w/ Rank Loss')
    plt.plot([12174.61] * 800, color='black')
    plt.xlabel('Offline Episodes')
    plt.ylabel('Average Reward')
    plt.title('Medium Replay vs Medium Expert World model for HalfCheetah')
    plt.legend()
    plt.savefig('cheetah_rep_vs_exp')
    plt.show()

def plotHopper():
    # # ml + rank
    # hopperMed1 = pd.read_csv(os.path.join('data', 'hopper_test1', 'hopper-medium-v2_fixed0_resid0_ModelBased_1_offline.csv'))
    # hopperMed2 = pd.read_csv(os.path.join('data', 'hopper_test1', 'hopper-medium-v2_fixed0_resid0_ModelBased_2_offline.csv'))
    # hopperMed4 = pd.read_csv(os.path.join('data', 'hopper_test1', 'hopper-medium-v2_fixed0_resid0_ModelBased_4_offline.csv'))
    
    # ml
    hopperRep5 = pd.read_csv(os.path.join('data', 'hopper_test1_ml', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_5_offline.csv'))
    hopperRep6 = pd.read_csv(os.path.join('data', 'hopper_test1_ml', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_6_offline.csv'))
    hopperRep7 = pd.read_csv(os.path.join('data', 'hopper_test1_ml', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_7_offline.csv'))
    idx = min(len(hopperRep5), len(hopperRep6), len(hopperRep7))
    hopperRepML = np.stack((
        np.array(hopperRep5['Reward'][:idx]),
        np.array(hopperRep6['Reward'][:idx]),
        np.array(hopperRep7['Reward'][:idx]),
    ),axis=0)
    rewardsRepML = np.mean(hopperRepML, axis=0) 
    rewardsStdRepML = np.std(hopperRepML, axis=0) 
    print('ml, max:', np.mean([max(hopperRep6['Reward']), max(hopperRep7['Reward'])]))
    print('ml, std:', np.std([max(hopperRep6['Reward']), max(hopperRep7['Reward'])]))

    # replay, ml + rank
    hopperRep3 = pd.read_csv(os.path.join('data', 'hopper_test1', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_3_offline.csv'))
    hopperRep5 = pd.read_csv(os.path.join('data', 'hopper_test1', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_5_offline.csv'))
    idx = min(len(hopperRep3), len(hopperRep5))
    hopperRep = np.stack((
        np.array(hopperRep3['Reward'][:idx]),
        np.array(hopperRep5['Reward'][:idx]),
    ),axis=0)
    rewardsRep = np.mean(hopperRep, axis=0) 
    rewardsStdRep = np.std(hopperRep, axis=0) 
    print('ml + rank, max:', np.mean([max(hopperRep3['Reward']), max(hopperRep5['Reward'])]))
    print('ml + rank, std:', np.std([max(hopperRep3['Reward']), max(hopperRep5['Reward'])]))

    # replay, rank
    hopperRep1 = pd.read_csv(os.path.join('data', 'hopper_test1_rank', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_1_offline.csv'))
    hopperRep2 = pd.read_csv(os.path.join('data', 'hopper_test1_rank', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_2_offline.csv'))
    hopperRep3 = pd.read_csv(os.path.join('data', 'hopper_test1_rank', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_3_offline.csv'))
    hopperRep4 = pd.read_csv(os.path.join('data', 'hopper_test1_rank', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_4_offline.csv'))
    hopperRep5 = pd.read_csv(os.path.join('data', 'hopper_test1_rank', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_5_offline.csv'))
    hopperRep6 = pd.read_csv(os.path.join('data', 'hopper_test1_rank', 'hopper-medium-replay-v2_fixed0_resid0_ModelBased_6_offline.csv'))
    idx = min(len(hopperRep2), len(hopperRep3), len(hopperRep4), len(hopperRep5), len(hopperRep6))
    hopperRank = np.stack((
        # np.array(hopperRep1['Reward'][:idx]),
        np.array(hopperRep2['Reward'][:idx]),
        np.array(hopperRep3['Reward'][:idx]),
        np.array(hopperRep4['Reward'][:idx]),
        np.array(hopperRep5['Reward'][:idx]),
        np.array(hopperRep6['Reward'][:idx]),
    ), axis=0)
    rewards = np.mean(hopperRank, axis=0) 
    rewardsStd = np.std(hopperRank, axis=0) 
    print('rank, max:', np.mean([max(hopperRep2['Reward']), max(hopperRep3['Reward']), max(hopperRep4['Reward']), max(hopperRep5['Reward']), max(hopperRep6['Reward'])]))
    print('rank, std:', np.std([max(hopperRep2['Reward']), max(hopperRep3['Reward']), max(hopperRep4['Reward']), max(hopperRep5['Reward']), max(hopperRep6['Reward'])]))
    
    hopperExp1 = pd.read_csv(os.path.join('data', 'hopper_test1_medexp_rank', 'hopper-medium-expert-v2_fixed0_resid0_ModelBased_1_offline.csv'))
    hopperExp2 = pd.read_csv(os.path.join('data', 'hopper_test1_medexp_rank', 'hopper-medium-expert-v2_fixed0_resid0_ModelBased_2_offline.csv'))
    hopperExp3 = pd.read_csv(os.path.join('data', 'hopper_test1_medexp_rank', 'hopper-medium-expert-v2_fixed0_resid0_ModelBased_3_offline.csv'))
    hopperExp4 = pd.read_csv(os.path.join('data', 'hopper_test1_medexp_rank', 'hopper-medium-expert-v2_fixed0_resid0_ModelBased_4_offline.csv'))
    idx = min(len(hopperExp1), len(hopperExp2), len(hopperExp3), len(hopperExp4))
    cheetahExp = np.stack((
        np.array(hopperExp1['Reward'][:idx]),
        np.array(hopperExp2['Reward'][:idx]),
        np.array(hopperExp3['Reward'][:idx]),
        np.array(hopperExp4['Reward'][:idx]),
    ), axis=0)
    rewardsExp = np.mean(cheetahExp, axis=0)
    # print(np.argmax(hopperExp1['Reward']), np.argmax(hopperExp2['Reward']), np.argmax(hopperExp3['Reward']), np.argmax(hopperExp4['Reward']))
    print('exp rank, max:', np.mean([max(hopperExp1['Reward']), max(hopperExp2['Reward']), max(hopperExp3['Reward']), max(hopperExp4['Reward'])]))
    print('exp rank, std:', np.std([max(hopperExp1['Reward']), max(hopperExp2['Reward']), max(hopperExp3['Reward']), max(hopperExp4['Reward'])]))

    plt.plot(np.arange(0, len(rewardsRepML), 5), rewardsRepML[0::5], color='blue', label='Medium Replay w/ ML Loss')
    plt.fill_between(range(len(rewardsRepML)), rewardsRepML - rewardsStdRepML , rewardsRepML + rewardsStdRepML, color='violet', alpha=0.2)
    
    plt.plot(np.arange(0, len(rewardsRep), 5), rewardsRep[0::5], color='orange', label='Medium Replay w/ ML + Rank')
    plt.fill_between(range(len(rewardsRep)), rewardsRep - rewardsStdRep , rewardsRep + rewardsStdRep, color='orange', alpha=0.2)
    
    plt.plot(np.arange(0, len(rewards), 5), rewards[0::5], color='green', label='Medium Replay w/ Rank Loss')
    plt.fill_between(range(len(rewards)), rewards - rewardsStd, rewards + rewardsStd, color='green', alpha=0.2)
    # plt.plot(np.arange(0, len(rewards), 5), [np.mean(rewards[i-5:i]) for i in range(len(rewards))][0::5], color='green', label='Medium Replay w/ Rank Loss')
    plt.plot([3512.09] * 800, color='black')
    plt.xlabel('Offline Episodes')
    plt.ylabel('Average Reward')
    plt.title('Hopper')
    plt.legend()
    plt.savefig('hopper')
    plt.show()
    
    plt.plot(np.arange(0, len(rewardsExp), 5), rewardsExp[0::5], label='Medium Expert w/ Rank Loss')
    plt.plot(np.arange(0, len(rewards), 5), rewards[0::5], label='Medium Replay w/ Rank Loss')
    plt.plot([3512.09] * 800, color='black')
    plt.xlabel('Offline Episodes')
    plt.ylabel('Average Reward')
    plt.title('Medium Replay vs Medium Expert World model for Hopper')
    plt.legend()
    plt.savefig('hopper_rep_vs_exp')
    plt.show()
 
plotCheetah()
print()
plotHopper()
