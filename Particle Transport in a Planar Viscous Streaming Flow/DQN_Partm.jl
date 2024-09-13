using NBInclude
using FileIO

@nbinclude("Ankh.ipynb")

log_filename = "log_loss_diff"
open(string(log_filename,".txt"), "w") do file
end

mutable struct initialise{T<:Number, F<:Number}
    mem_size::T;
    discrete::Bool;
    mem_counter::T;
    state_memory::Array{F};
    newstate_memory::Array{F};
    action_memory::Array{F};
    reward_memory::Array{F};
    terminal_memory::Array{F};
end

#abstract type dqn end

mutable struct agent_initialise{T1<:Number, D1<: Number}
    action_space::Array{T1}
    gamma::D1
    epsilon::D1
    epsilon_decay::D1
    epsilon_end::D1
    alpha::D1
    batch_size::T1
end

const max_size = 1000000
const input_shape = 2
const n_actions = 16
const discrete = true

function initialisation()
    data = initialise(max_size, discrete, 0, zeros(max_size, input_shape), 
        zeros(max_size, input_shape), zeros(max_size, n_actions), zeros(max_size), zeros(max_size) )
    agentdata = agent_initialise([i for i in 1:n_actions], 0.99, 1.0, 0.99, 0.01, 0.001, 64)
    return data, agentdata
end    
data, agentdata = initialisation()

function transitionstore(state, action, reward, state_n, done)
    ind = (data.mem_counter % data.mem_size)+1
    data.state_memory[ind,:] .= state
    data.newstate_memory[ind,:] .= state_n
    data.reward_memory[ind,:] .= reward
    data.terminal_memory[ind,:] .= 1 - Int8(done)
    actions = zeros(length(data.action_memory[1,:]))
    actions[action] = 1.0
    data.action_memory[ind,:] .= actions
    data.mem_counter += 1
end 

function sampling(batchsize)
    max_mem = min(data.mem_counter, data.mem_size)
    batch = rand(1:max_mem, batchsize)
    bstate = data.state_memory[batch,:]
    bstate_new = data.newstate_memory[batch,:]
    breward = data.reward_memory[batch,:]
    baction = data.action_memory[batch,:]
    bterminal = data.terminal_memory[batch,:]
    return bstate, baction, breward, bstate_new, bterminal
end

function remember(state, action, reward, new_state, done)
    transitionstore(state, action, reward, new_state, done)
end

function action_choice(state)
    #println(state)
    if rand() < agentdata.epsilon
        action = rand(agentdata.action_space)
    else
        actions,_ = forward_propagation([state[1] state[2]], nn_param.param_learn, nn_param.activation_type)
        action = argmax(actions)[2] 
    end
    return action
end

function learn()
        if data.mem_counter < agentdata.batch_size
            return
        end
        state, action, reward, new_state, done = sampling(agentdata.batch_size)                                                      
        action_values = reshape(agentdata.action_space,(n_actions,1))
        action_indices = sum(action .* action_values', dims=2)  
    
        q_eval,_ = forward_propagation(state, nn_param.param_learn, nn_param.activation_type)
        q_next,_ = forward_propagation(new_state, nn_param.param_learn, nn_param.activation_type)
        
        #q_eval = transpose(model_predict(transpose(state)))
        #q_next = transpose(model_predict(transpose(new_state)))       
        q_target = deepcopy(q_eval)
        
        batch_index = reshape([i for i in 1:agentdata.batch_size],(agentdata.batch_size,1))
        
        #CartInd1 = []
        #for i in 1:agentdata.batch_size
        #    append!(CartInd,[(i,action_indices[i,1])])
        #end

        CartIndx = [CartesianIndex(batch_index[i,1],Int64.(action_indices[i,1])) for i in 1:length(batch_index)]
        #println(size(CartIndx))
        q_target[CartIndx] = (reward + agentdata.gamma*(maximum(q_next, dims = 2).*done))  
     
        
        nn_param.param_learn = train_model(copy(nn_param.param_learn), state, q_target, nn_param.layers, 
                                nn_param.activation_type, nn_param.loss_fcn;lr = 0.0001,epochs = 20,
                                verbose = false, optimiser = "Adam", shuffle_data = true,lambda = 0.0, n_reg = 2,
                                steps_per_epoch = 2, batch_size = nothing,log = false, log_file = log_filename)
        
        #opt = Descent(agentdata.alpha)
        #parameters = Flux.params(model_predict)
        #loss(x,y) = Flux.Losses.mse(model_predict(transpose(x)),transpose(y))
        #data1 = [(state, q_target)]
        
        #train!(loss,parameters,data1,opt)
        
        if agentdata.epsilon > agentdata.epsilon_end
           agentdata.epsilon = agentdata.epsilon*agentdata.epsilon_decay
        
        else agentdata.epsilon_end 
        end
end

function savemodel(arg)
    save(string(arg,".jld2"), "data", nn_param.param_learn)
end

function loadmodel(arg)
    return load(string(arg,".jld2"))["data"];
end

mutable struct parameters_learn{T<:Dict{Any, Any}, B<:Vector{String}, D<:Vector{Int64},P<:String}
    param_learn::T
    activation_type::B
    layers::D
    loss_fcn::P
end

nn_param = parameters_learn(initialise_parameters([input_shape,128,128, 128, n_actions], "he_uniform"),["ReLu","ReLu", "ReLu","Linear"],[input_shape,128,128,128, n_actions],"mse")

nn_param.param_learn


