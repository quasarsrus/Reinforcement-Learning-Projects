using Pkg
using Plots
using ViscousStreaming
using NBInclude
using HDF5
using JLD2
using Distributions

#Pkg.update()

#Pkg.add("HDF5")
#Pkg.add("Distributions")
#Pkg.add("JLD2")

#@nbinclude("strdvelx.ipynb")

strdVuxy = load("strdVuxy_square.jld2")["data"];
strdVvxy = load("strdVvxy_square.jld2")["data"];

 Re = 40
 ϵ = 0.1
 Ω = 1.0 # frequency (keep this equal to 1)
 Tp = 2π/Ω # one period of oscillation
 p = StreamingParams(ϵ,Re)
 #s = StreamingAnalytical(p)

 τ = 0.1 # Stokes number, should be small
 β = 0.95 # Density parameter. Less than 1 means heavier than fluid.
 p_inert = InertialParameters(tau=τ,beta=β,epsilon=ϵ,Re=Re)
 Ω = 1.0
 Tp = 2π/Ω
 Tmax = 50*Tp

 Δx = 0.02
 xlim = (-2.8,2.8)
 ylim = (-2.8,2.8)
 n = 75
 body = Circle(0.2,n)

bl = BodyList()
bL1 = deepcopy(body)
bL2 = deepcopy(body)
bR1 = deepcopy(body)
bR2 = deepcopy(body)

# left 1 cylinder
cent = (-2.0,2.0)
α = 0.0
TL = RigidTransform(cent,α)
TL(bL1) # transform the body to the current configuration

# left 2 cylinder
cent = (-2.0,-2.0)
α = 0.0
TL = RigidTransform(cent,α)
TL(bL2) # transform the body to the current configuration

# right 1 cylinder
cent = (2.0,2.0)
α = 0.0
TR = RigidTransform(cent,α)
TR(bR1) # transform the body to the current configuration

# right 2 cylinder
cent = (2.0,-2.0)
α = 0.0
TR = RigidTransform(cent,α)
TR(bR2) # transform the body to the current configuration

push!(bl,bL1);
push!(bl,bL2);
push!(bl,bR1);
push!(bl,bR2);

function mean_motion(dR,R,p,t,v̄Luxy,v̄Lvxy)
    dR[1] = v̄Luxy(R[1],R[2])
    dR[2] = v̄Lvxy(R[1],R[2])
   return dR 
end
   
function reset1()
    px = rand(Uniform(-1.8,1.8))
    py = rand(Uniform(-1.8,1.8))
    return (px,py)
end

function motion(curstate, ampvel_ind)
    newposx = 0
    newposy = 0
    done = false
    
    if ampvel_ind == 16
        newposx, newposy = curstate
    else
        
        v̄Lfcn(dR,R,p,t) = mean_motion(dR,R,p,t,strdVuxy[ampvel_ind],strdVvxy[ampvel_ind])
        solL = compute_trajectory(v̄Lfcn,curstate,Tmax,10Tp,bl=bl,ϵ=p.ϵ);
        newposx = last(solL[1,:])
        newposy = last(solL[2,:])
    end
    
    if newposx < -1.8  || newposx > 1.8  || newposy < -1.8  || newposy > 1.8 
        reward = -50
        newposx,newposy = curstate
    elseif newposx < 0.1  && newposx > -0.1  && newposy < 0.1  && newposy > -0.1
        reward = 2 - (abs(newposx) + abs(newposy)) 
        if newposx < 0.02  && newposx > -0.02  && newposy < 0.02  && newposy > -0.02 
            reward = 1000
            done = true
        end
    else
        reward = -1
    end
    """
    if newposx < -1.8  || newposx > 1.8  || newposy < -1.8  || newposy > 1.8 
        reward = -50
        newposx,newposy = curstate
    elseif newposx < 1.1  && newposx > 0.9  && newposy < 0.1  && newposy > -0.1
        reward = 2 - (abs(1-newposx) + abs(newposy-0)) 
        if newposx < 1.02  && newposx > 0.98  && newposy < 0.02  && newposy > -0.02 
            reward = 1000
            done = true
        end
    else
        reward = -1
    end
    """
    """
    if newposx < -1.8  || newposx > 1.8  || newposy < -1.8  || newposy > 1.8 
        reward = -50
        newposx,newposy = curstate
    elseif newposx < -0.9  && newposx > -1.1  && newposy < 0.1  && newposy > -0.1
        reward = 2 - (abs(1+newposx) + abs(newposy)) 
        if newposx < -0.98  && newposx > -1.02  && newposy < 0.02  && newposy > -0.02 
            reward = 1000
            done = true
        end
    else
        reward = -1
    end
    """

    return (newposx,newposy), reward, done
end


function main()
    scores = []
    episode_history = []
    episodes = 1000

    for i in 1:episodes
        done = false
        score = 0
        obs = reset1()
        j = 0
        while !done
        #for j = 1:200
            action = action_choice(obs,nn_param) 
            obs_new, reward, done = motion(obs, action)
            score += reward 
            remember(obs, action, reward, obs_new, done)
            obs = obs_new
            learn(nn_param, nn_param_target)
            j += 1
            if j == 200
              break
            end
        end
                
        append!(episode_history,agentdata.epsilon)
        append!(scores,score)
        
        avgscore = mean(scores[max(1, i-100):(i)])
        #if i%50 == 0
            println("score -> $score, episode -> $i, Average Score -> $avgscore")
        #end
    end
        #if i%10 == 0 and i>0:
        #    savemodel()
end      


@nbinclude("DQN_Partm.ipynb")

@nbinclude("DDQN_MultiPartM.ipynb")

nn_param = model_init("");
nn_param_target = deepcopy(nn_param);

nn_param.param_learn = loadmodel("Weights_Partm(centre)")

@time main()

savemodel("Weights_Partm(Left)")

@time solver2 = FrequencyStreaming(Re,ϵ,Δx,xlim,ylim,bl);

function motion_exploit(curstate, ampvel_ind)
    if ampvel_ind == 16
        return curstate
    else
        v̄Lfcn(dR,R,p,t) = mean_motion(dR,R,p,t,strdVuxy[ampvel_ind],strdVvxy[ampvel_ind])
        solL = compute_trajectory(v̄Lfcn,curstate,Tmax,10Tp,bl=bl,ϵ=p.ϵ);
    end
    return [solL[1,:],solL[2,:]]
end

function policyrollout(obs)
    pos7 = []
    pos8 = []
    done = false
    agentdata.epsilon = 1
    push!(pos7, obs[1])
    push!(pos8, obs[2])
    action_chosen_ind = []
    push!(action_chosen_ind,16)
    for i = 1:50 
        actions,_ = forward_propagation([obs[1] obs[2]], nn_param.param_learn, nn_param.activation_type)
        action = argmax(actions)[2]
        obs_new = motion_exploit(obs, action)
        for j = 1:length(obs_new[1])
            push!(pos7, obs_new[1][j])
            push!(pos8, obs_new[2][j])
        end
        for j = 1:length(obs_new[1])
            push!(action_chosen_ind,action)
        end

        obs = (last(obs_new[1]),last(obs_new[2]))
        
    end
    return pos7, pos8, action_chosen_ind
end

function policyrollout_biased(obs)
    pos7 = []
    pos8 = []
    push!(pos7, obs[1])
    push!(pos8, obs[2])
    action_chosen_ind = []
    push!(action_chosen_ind,16)
    action = 1
    for i = 1:300 
        obs_new = motion_exploit(obs, action)
        for j = 1:length(obs_new[1])
            push!(pos7, obs_new[1][j])
            push!(pos8, obs_new[2][j])
            push!(action_chosen_ind,action)
        end
        
        if i > 150
            action = 7;
        end

        obs = (last(obs_new[1]),last(obs_new[2]))
        
    end
    return pos7, pos8, action_chosen_ind
end

pos7, pos8, act_ind = policyrollout(reset1());

pos7, pos8, act_ind = policyrollout_biased(reset1());

soln = solver2([ComplexF64[0.0,0.0] for i in 1:length(bl)], bl);
xg, yg = coordinates(soln.s1.W,solver2.grid)
plot(pos7, pos8, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot!(xg,yg,vorticity(0,soln.s1) , levels=range(-2,2,length=30),clim=(-2,2),xlim=xlim,ylim=ylim)
plot!(bl)

@time @gif for k = 1:length(pos7)
    soln = solver2([ComplexF64[0.0,0.0] for i in 1:length(bl)], bl);
    xg, yg = coordinates(soln.s1.W,solver2.grid)
    plot(pos7[1:k], pos8[1:k], ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    plot!(bl)
    #scatter!([2.0,-2.0,-2.0,2.0],[2.0,2.0,-2.0,-2.0] , mc=:lightgreen, ms=14)
    if act_ind[k] != 16
        scatter!(cyl_type_x[act_ind[k]],cyl_type_y[act_ind[k]] , mc=:blue, ms=14)
    end
end

anim =  @animate for k = 1:length(pos7)   
    soln = solver2([ComplexF64[0.0,0.0] for i in 1:length(bl)], bl);
    xg, yg = coordinates(soln.s1.W,solver2.grid)
    #plot(pos1, pos2, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    #plot!(pos3, pos4, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    #plot!(pos5, pos6, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    plot(pos7[1:k], pos8[1:k], ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    #plot!(xg,yg,vorticity(0,soln.s1) , levels=range(-2,2,length=30),clim=(-2,2),xlim=xlim,ylim=ylim)
    #plot!(bl)
    scatter!([2.0,-2.0,-2.0,2.0],[2.0,2.0,-2.0,-2.0] , mc=:lightgreen, ms=14)
    if act_ind[k] != 16
        scatter!(cyl_type_x[act_ind[k]],cyl_type_y[act_ind[k]] , mc=:red, ms=14)
    end
    
end

gif(anim,fps=2)

cyl_type_x = [[-2.0],[-2.0,-2.0],[-2.0,-2.0,2.0],[-2.0,-2.0,2.0,2.0],[-2.0,2.0,2.0],[2.0,2.0],[2.0],[-2.0,2.0],[-2.0,-2.0,2.0],
[-2.0,2.0],[-2.0,2.0,2.0],[-2.0,2.0],[2.0],[-2.0,2.0],[-2.0]
]
cyl_type_y = [[2.0],[2.0,-2.0],[2.0,-2.0,2.0],[2.0,-2.0,2.0,-2.0],[-2.0,2.0,-2.0],[2.0,-2.0],[-2.0],[-2.0,-2.0],
    [2.0,-2.0,-2.0],[2.0,-2.0],[2.0,2.0,-2.0],[2.0,2.0],[2.0],[-2.0,2.0],[-2.0]
]

function policyrollout(obs1,obs2)
    agentdata.epsilon = 1
    posa = [] 
    posb = [] 
    posc = [] 
    posd = []
    done1 = false
    done2 = false
    push!(posa, obs1[1])
    push!(posb, obs1[2])
    push!(posc, obs2[1])
    push!(posd, obs2[2])
    for i = 1:10000
        if i%2 == 0
            actions,_ = forward_propagation([obs1[1] obs1[2]], nn_param.param_learn, nn_param.activation_type)
        else 
            actions,_ = forward_propagation([obs2[1] obs2[2]], nn_param.param_learn, nn_param.activation_type)
        end
        action = argmax(actions)[2]
        obs1_new, reward, done1 = motion(obs1, action)
        obs2_new, reward, done2 = motion(obs2, action)
        push!(posa, obs1[1])
        push!(posb, obs1[2])
        push!(posc, obs2[1])
        push!(posd, obs2[2])
        obs1 = obs1_new
        obs2 = obs2_new  
    end
    return posa,posb,posc,posd
end

posa, posb, posc, posd = policyrollout(reset1(),reset1());

left = loadmodel("Weights_Partm(Left)")
right = loadmodel("Weights_Partm(right)")

function policyrollout_diff(obs1,obs2)
    agentdata.epsilon = 1
    posa = [] 
    posb = [] 
    posc = [] 
    posd = []
    done1 = false
    done2 = false
    push!(posa, obs1[1])
    push!(posb, obs1[2])
    push!(posc, obs2[1])
    push!(posd, obs2[2])
    for i = 1:50000
        if i%2 == 0
            actions,_ = forward_propagation([obs1[1] obs1[2]], left, nn_param.activation_type)
        else 
            actions,_ = forward_propagation([obs2[1] obs2[2]], right, nn_param.activation_type)
        end
        action = argmax(actions)[2]
        obs1_new, reward, done1 = motion(obs1, action)
        obs2_new, reward, done2 = motion(obs2, action)
        push!(posa, obs1[1])
        push!(posb, obs1[2])
        push!(posc, obs2[1])
        push!(posd, obs2[2])
        obs1 = obs1_new
        obs2 = obs2_new  
    end
    return posa,posb,posc,posd
end

posa, posb, posc, posd = policyrollout_diff(reset1(),reset1());

soln = solver2([ComplexF64[0.0,0.0] for i in 1:length(bl)], bl);
xg, yg = coordinates(soln.s1.W,solver2.grid)
#plot(pos1, pos2, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
#plot!(pos3, pos4, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot(posa, posb, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot!(posc, posd, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot!(xg,yg,vorticity(0,soln.s1) , levels=range(-2,2,length=30),clim=(-2,2),xlim=xlim,ylim=ylim)
plot!(bl)

@gif for k = 1:length(posa)
    soln = solver2([ComplexF64[0.0,0.0] for i in 1:length(bl)], bl);
    xg, yg = coordinates(soln.s1.W,solver2.grid)
    #plot(pos1, pos2, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    #plot!(pos3, pos4, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    plot(posa[1:k], posb[1:k], ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    plot!(posc[1:k], posd[1:k], ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    plot!(xg,yg,vorticity(0,soln.s1) , levels=range(-2,2,length=30),clim=(-2,2),xlim=xlim,ylim=ylim)
    plot!(bl)
end

using simplefunc

using Ankh

function policyrollout_trial(obs1,obs2)
    agentdata.epsilon = 1
    posa = [] 
    posb = [] 
    posc = [] 
    posd = []
    done1 = false
    done2 = false
    push!(posa, obs1[1])
    push!(posb, obs1[2])
    push!(posc, obs2[1])
    push!(posd, obs2[2])
    for i = 1:100
        actions,_ = forward_propagation([obs1[1] obs1[2]], nn_param.param_learn, nn_param.activation_type)
        action = argmax(actions)[2]
        obs1_new, reward, done1 = motion(obs1, action)
        obs2_new, reward, done2 = motion(obs2, action)
        push!(posa, obs1[1])
        push!(posb, obs1[2])
        push!(posc, obs2[1])
        push!(posd, obs2[2])
        obs1 = obs1_new
        obs2 = obs2_new  
        if(done1)
            break
        end
    end
    return posa,posb,posc,posd
end

pos_test = reset1()
pos_test_2 = (pos_test[1]+0.1,pos_test[2]-0.1)

posa, posb, posc, posd = policyrollout_trial(pos_test,pos_test_2);

soln = solver2([ComplexF64[0.0,0.0] for i in 1:length(bl)], bl);
xg, yg = coordinates(soln.s1.W,solver2.grid)
#plot(pos1, pos2, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
#plot!(pos3, pos4, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot(posa, posb, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot!(posc, posd, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot!(xg,yg,vorticity(0,soln.s1) , levels=range(-2,2,length=30),clim=(-2,2),xlim=xlim,ylim=ylim)
plot!(bl)

soln = solver2(amp,body);

plot(streamfunction(0,soln.s1),soln.g,levels=range(-1,1,length=31),xlim=xlim,ylim=ylim)
plot!(body)

function policyrollout_diff_test(obs1,obs2)
    agentdata.epsilon = 1
    posa = [] 
    posb = [] 
    posc = [] 
    posd = []
    done1 = false
    done2 = false
    push!(posa, obs1[1])
    push!(posb, obs1[2])
    push!(posc, obs2[1])
    push!(posd, obs2[2])
    v̄Lfcn(dR,R,p,t) = mean_motion(dR,R,p,t,strdVuxy[1],strdVvxy[1])
    solL_dummy1 = compute_trajectory(v̄Lfcn,obs1,35000*Tp,10Tp,bl=bl,ϵ=p.ϵ);
    #obs1 = (last(solL_dummy[1,:]),last(solL_dummy[2,:]))
    solL_dummy2 = compute_trajectory(v̄Lfcn,obs2,35000*Tp,10Tp,bl=bl,ϵ=p.ϵ);
    #obs2 = (last(solL_dummy[1,:]),last(solL_dummy[2,:]))
    
    for i = 1:length(solL_dummy1[1,:])
        
        obs1 = (solL_dummy1[1,i],solL_dummy1[2,i])
        obs2 = (solL_dummy2[1,i],solL_dummy2[2,i])
        
        push!(posa, obs1[1])
        push!(posb, obs1[2])
        push!(posc, obs2[1])
        push!(posd, obs2[2])
   
    end

    for i = 1:50
        #if i%2 == 0
            actions,_ = forward_propagation([obs1[1] obs1[2]], nn_param.param_learn, nn_param.activation_type)
        #else 
            #actions,_ = forward_propagation([obs2[1] obs2[2]], nn_param.param_learn, nn_param.activation_type)
        #end
        action = argmax(actions)[2]
        obs1_new, _, done1 = motion(obs1, action)
        obs2_new, _, done2 = motion(obs2, action)
        push!(posa, obs1[1])
        push!(posb, obs1[2])
        push!(posc, obs2[1])
        push!(posd, obs2[2])
        obs1 = obs1_new
        obs2 = obs2_new  
        if(done1 || done2)
            break
        end
    end
    return posa,posb,posc,posd
end

posa, posb, posc, posd = policyrollout_diff_test(reset1(),reset1());

soln = solver2([ComplexF64[0.0,0.0] for i in 1:length(bl)], bl);
xg, yg = coordinates(soln.s1.W,solver2.grid)
#plot(pos1, pos2, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
#plot!(pos3, pos4, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot(posa, posb, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot!(posc, posd, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
plot!(xg,yg,vorticity(0,soln.s1) , levels=range(-2,2,length=30),clim=(-2,2),xlim=xlim,ylim=ylim)
plot!(bl)

@gif for k = 1:length(posa)
    soln = solver2([ComplexF64[0.0,0.0] for i in 1:length(bl)], bl);
    xg, yg = coordinates(soln.s1.W,solver2.grid)
    #plot(pos1, pos2, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    #plot!(pos3, pos4, ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    plot(posa[1:k], posb[1:k], ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    plot!(posc[1:k], posd[1:k], ratio=1,legend=false,linewidth=1,xlim=xlim,ylim=ylim)
    plot!(xg,yg,vorticity(0,soln.s1) , levels=range(-2,2,length=30),clim=(-2,2),xlim=xlim,ylim=ylim)
    plot!(bl)
end


