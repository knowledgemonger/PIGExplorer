% explorer is a class for creating and training explorer networks
% to use this classdef it must be place in a folder called
% "@explorer" in the local directory
% an explorer can be constructed as:
%          >> variable_name=explorer(input)
% where input equals ...

classdef explorer< handle
  properties
    M % number of actions
    N % number of states
    MN % M*N
    P % [MxNxN] true CMC kernel
    Q % [MxNxNxK] internal model of CMC kernel for strategy K
    Mask % [MxNxN] one if P(a,s,s')>0, 0 otherwise
    strategies % list of strategies tested
    T % length of trial in time-steps
    A_history % [TxK] history of actions taken
    S_history % [TxK] history of states reached
    KL_history % [TxK] history of KL divergences between P and Q
    nav_dist % [TxNxK] epected pathlength under Q
    Q_nav_dist
    
    nav_cutoff % max number of time-steps allowed to reach target in nav trials
    obj_navdist % [1xN] expected pathlength under P
    obj_lost % [1xN] nav_lost under P
    lambda % concentration parameter for generative Dirichlet distribution
    context % for inference with mixed priors
    infer_lambda % boolean saying to infer lambda by maximum liklihood estimate
    testnav
    discount
    contextual_inference_history
    PI_history
    opt_nav_dist
    rew_cutoff
    U_index
    U_history
    
    U2_history
    PImeanstd
    
    IGmeanstd
    PI_chosen_history
    PIvsPIG_index
    PI_index
    PI_gain
    PI_gain2
    rew_gains
    opt_rew_gains
    PI_calculated_history
    PI_sampled
    Ay_pol_pos
    Ay_pol
    Q_store
    Qbeta=0.5; % forgetfullness (0=no update, 1=fully replace each update)
    Qgamma=0.9; % discount factor
    QPIbeta;
    QPIgamma;
    Qnavbeta=0.2;
    Qnavgamma=0.7;
    
    Value_history
    VI_index
    Q_history
    U_store
    U2_store
    epsilon
    avgU_history
    avgU2_history
    
    PI2_history
    Q2_store
    Q2_history
    U3_store
    U4_store
    U4_history
    avgU4_history
  end
  
  methods (Access = 'private')
    
    
  end
  
  methods (Access = 'public')
    
    
    function obj=explorer(P)
      obj.P=P;
      obj.M=size(P,1);
      obj.N=size(P,2);
      obj.MN=obj.M*obj.N;
    end
    
    
    function explore (obj, strategies, T, lambda, Mask, context,discount,PI_gain, nav_cutoff,rew_cutoff, reward_trials)
      % Initialize variables
      N=obj.N;
      M=obj.M;
      MN=obj.MN;
      K=length(strategies);
      zerosMN=zeros(M,N);
      zerosN1=zeros(N,1);
      zeros11N=zeros(1,1,N);
      zerosM1=zeros(M,1);
      zerosNN=zeros(N,N);
      eyeN=eye(N);
      
      % Transition Distributions
      P=obj.P;                                    % P = objective transition probabilities [M x N x N] (conditional A, condition S, outcome S)
      if any(size(Mask)~=[M,N,N])
        error(0,'Mask size does not match');
      end
      if any(Mask(:)~=0)~=1
        error(0,'Mask must have only 0 and 1 entries');
      end
      if any(P(Mask(:)==0)~=0)
        error(0,'P contains non-0 non-masked elements');
      end
      
      Q=ones(M,N,N,K);                          % Q = subjective transition probabilities [M x N x N x K] (conditional A, condition S, outcome S, strategy)
      
      P_cumulation=cumsum (P,3);                  % P_cumulation = cumulative sum of P over outcomes [M x N x !N!]
      if K==1
        Krep_P=P;                                % Krep_P = repetition of P over strategies [M x N x N x !K!]
        KMask=Mask;
      else
        Krep_P=repmat(P,[1,1,1,K]);
        KMask=repmat(Mask,[1,1,1,K]);
      end
      OutSize=sum(Mask,3);
      KOutSize=repmat(OutSize,[1,1,K]);
      Q=bsxfun(@rdivide,Q.*KMask,sum(Q.*KMask,3));
      
      Q_nav=zeros(M,N,N,K);
      
      pregen_rand=rand(T-1,1);                    % pregen_rand = pregenerated random numbers uniformly distributed from 0 to 1 [T-1 x 1] used for simulating transitions
      
      % Expedition Histories
      S=zeros(T,K)+NaN;                    % S = history of states [T x K]
      A=S;                                        % A = history of actions [T x K]
      S(1,:)=randsample(N,1);                                   % S initialized to state 1
      
      KL_history=zeros(T,K)+NaN;                    % KL_history = history of summed KL-divergences between elements of P and Q [T x K]
      KL=Krep_P.*log2(Krep_P./Q).*KMask;                 % KL = KL contribution of each parameter to KL-divergence between P and Q [M x N x N x K]
      KL(isnan(KL))=0;
      KL_history(1,:)=sum(sum(sum(KL,3),2),1);
      
      % Contextual Learning Variables
      infer_lambda=false;                         % infer_lambda = true-> lambda inference on, false -> constant lambda [boolean]
      if length(lambda)==1
        if lambda==0
          infer_lambda=true;
          options.numDiff=1;
          options.Display='off';
          maxlambda=30;
          Jlambda=ones(M,N,K)*maxlambda.*KOutSize;               % Jlambda = smoothing factor for Qs across conditionals [M x N x K]
          Wlambda=maxlambda.*KMask; % Wlambda = smoothing factor repeted across posterior states
          if length(context)==1                 % context = equilancy class associated with each state-action pair [M x N]
            context=ones(M,N);
          end
          number_contexts=max(context(:));      % number_contexts = number of equivilency classes [1]
          rep_context=repmat(context,[1,1,N]);  % rep_context = repeated context matrix for separating different expeditions [M x N x K]
          for c=1:number_contexts
            context_size(c)=sum(context(:)==c);%#ok<AGROW> % context_size = size of each equivalency class [1]
          end
          contextual_inference_history=zeros(T,number_contexts,K)+maxlambda;                % contextual_inference_history = history of inferred lambdas for each context [T x number_contexts x K]
          strat_lambda=zeros(M,N);              % strat_lambda = temporary holder for of inferred lambdas for current strategy
        else
          Jlambda=(zeros(M,N,K)+lambda).*KOutSize;
          Wlambda=lambda.*KMask;
          number_contexts=1;
        end
      elseif all(size(lambda)==[M,N])
        Jlambda=repmat(lambda,[1,1,K]).*KOutSize;
        Wlambda=repmat(lambda,[1,1,N,K]).*KMask;
        number_contexts=1;
      else
        error(0,'lambda not an appropriate size');
      end
      
      % State, Joint Conditional, and Full Transition Histograms and Distributions
      S_hist=zeros(N,K);                          % S_hist = histogram of conditional states [N x K]
      J_hist=zeros(M,N,K);                        % J_hist = joint histogram of visited states and enacted actions [M x N x K]
      S_dist=S_hist;                              % S_dist = Sampling distirbution on States [N x K]
      S_dist(:)=1/N;
      J_dist=J_hist;                              % J_dist = Sampling distribution on Conditionals [M x N x K]
      J_dist(:)=1/MN;
      W_hist=zeros(M,N,N,K);                      % W_hist = histogram of full transition events [M x N(conditional) x N(transition) x K]
      J_smoothed=J_hist+Jlambda;                % J_smoothed = smoothed histogram of joint conditionals [M x N x K]
      W_smoothed=W_hist+Wlambda;                  % W_smoothed = smoothed histogram of full transitions [M x N x N x K]
      
      S_hist_new=zeros(N,K);
      S_joint=zeros(N,N,K);
      
      Spair_prior=zeros(N*N,K);
      Spair_post=Spair_prior;
      Spair_joint=zeros(N*N,N*N,K);
      
     
      epsilon=[];
      
      %Qmu=0.1; %not used in current implementation, probability of doing
      %random action instead of Q-action
      Qbeta=obj.Qbeta; % forgetfullness (0=no update, 1=fully replace each update)
      Qgamma=obj.Qgamma; % discount factor
      
        tao=0.1;%0.1%0.0001; % soft max weight (0 = greedy, up = smooth
      
        obj.QPIbeta=0.1;
        obj.QPIgamma=0.9;
      QPIbeta=obj.QPIbeta; % forgetfullness for strats 23 24 25
      QPIgamma=obj.QPIgamma; % discount factor for strats 23 24 25
      Qnavbeta=obj.Qnavbeta; % forgetfullness for strats 23 24 25
      Qnavgamma=obj.Qnavgamma; % discount factor for strats 23 24 25
      
      PI_history=zeros(T,K);
      PI2_history=zeros(T,K);
      
      % BEGIN: Initialize values for First time step
      EdKL=zerosMN;
      
      for a=1:M
        for s=1:N
          projected_W_smoothed=bsxfun(@plus,squeeze(W_smoothed(a,s,:,1)),eyeN); % [N(future) x N(hypothesized outcome)]
          projected_W_smoothed(:,~Mask(a,s,:))=0;
          projected_Q=projected_W_smoothed/(J_smoothed(a,s,1)+1);
          
       
            
            EdKL(a,s)=nansum(projected_Q.*log2(bsxfun(@rdivide,projected_Q,squeeze(Q(a,s,:,1)))),1)*squeeze(Q(a,s,:,1)); %sum_s+ q(s+|D)*(sum_s p(s|D,s+)log2 (p(s|D,s+)/p(s|D)))
          psi_projected=psi(projected_W_smoothed);
          psi_projected(isinf(psi_projected))=0;
          psi_sum_projected=psi(nansum(projected_W_smoothed,1));
          psi_sum_projected(isinf(psi_sum_projected))=0;
          gamma_projected=gamma(projected_W_smoothed);
          gamma_projected(isinf(gamma_projected))=1;
          gamma_current=gamma(W_smoothed(a,s,:,1));
          gamma_current(isinf(gamma_current))=1;
          gamma_sum_projected=gamma(nansum(projected_W_smoothed,1));
          gamma_sum_projected(isinf(gamma_sum_projected))=1;
          gamma_sum_current=gamma(nansum(W_smoothed(a,s,:,1)));
          gamma_sum_current(isinf(gamma_sum_current))=1;
          
          EdhyperKL(a,s)=(nansum(bsxfun(@minus,projected_W_smoothed,squeeze(W_smoothed(a,s,:,1))).*bsxfun(@minus,psi_projected,psi_sum_projected),1)-nansum(log(gamma_projected),1)+nansum(log(gamma_current))+log(gamma_sum_projected/gamma_sum_current))*squeeze(Q(a,s,:,1));
        end
      end
      
      U_index=zeros(1,K);Q_index=U_index;PI_index=U_index;PIvsPIG_index=U_index;emp2_index=U_index;Ay_index=U_index;Ay2_index=U_index;PI_samp_index=U_index;VI_index=U_index;U3_index=U_index;Q2_index=U_index;

      for i=1:K
        k=strategies(i);
        if k==30
          pregen_random_a=randsample(M,T,true);    % pregenerated random action selection [1 x T]
        end
        if any(k==[1,2,3,4,5,13]) % Set these strategies utility function to initial PIG
          U_index(i)=max(U_index)+1;
          U_store(:,:,U_index(i))=EdKL; %#ok<AGROW>
        end
        if any(k==[3,4,8,9,13]) % Allocate memory for these strategies initial VI value
          VI_index(i)=max(VI_index)+1;
          Value_history(:,:,:,VI_index(i))=zeros(M,N,floor(T/50));
        end
        if any(k==[6,7,8,9,10]) % Set these strategies utility function to initial Surprise of 0
          U_index(i)=max(U_index)+1;
          U_store(:,:,U_index(i))=zerosMN; %#ok<AGROW>
        end
        if any(k==[13]) % For strategies that combine PI and PIG, allocate memory for storing which strategy was implemented at each time step
          PIvsPIG_index(i)=max(PIvsPIG_index)+1;
          PI_chosen_history(:,PIvsPIG_index(i))=zeros(T,1); %#ok<AGROW>
        end
        if any(k==[5,10]) % For strategies using Q-learning preallocate Q-learning values
          Q_index(i)=max(Q_index)+1;
          Q_store(:,:,Q_index(i))=log2(OutSize); %#ok<AGROW>
          Q_history(:,:,:,Q_index(i))=zeros(M,N,floor(T/50));
          epsilon(:,:,Q_index(i))=zerosMN;
        end
        if any(k==[11,12,13]) % For strategies that implement AY gradient descent on PI
          Ay_index(i)=max(Ay_index)+1;
          Ay_pol(:,:,Ay_index(i))=ones(M,N)/M; %#ok<AGROW>
          S_dist_hat(:,Ay_index(i))=ones(N,1)/N; %#ok<AGROW>
          F(:,:,Ay_index(i))=zeros(M,N,1); %#ok<AGROW>
          
          PI_index(i)=max(PI_index)+1;
          PI_store(PI_index(i))=0; %#ok<AGROW>
        end
        
%         if k==12 % Calculate the PI strategy optimized by gradient descent on PI using true world structure
%           converged=false;
%           S_hist_pos=ones(N,1)*1000;
%           S_dist_hat_pos=ones(N,1)/N;
%           s2=randsample(N,1);
%           Ay_pol_pos=ones(M,N)/M;
%           t=1;
%           S_hist_pos(s2)=S_hist_pos(s2)+1;
%           while ~converged
%             t=t+1;
%             s1=s2;
%             Ay_pol_old=Ay_pol_pos;
%             a=randsample(M,1,true,Ay_pol_pos(:,s1));
%             s2=sum(rand(1)>P_cumulation(a,s,:))+1;
%             S_hist_pos(s2)=S_hist_pos(s2)+1;
%             S_dist_hat_pos=S_hist_pos/(t+1000*N);
%             %             S_dist_hat_pos=(t^0.5*S_dist_hat_pos)/(t^0.5+1); %#ok<AGROW>
%             %             S_dist_hat_pos(s2)=S_dist_hat_pos(s2)+1/(t^0.5+1); %#ok<AGROW>
%             sprime_on_s=sum(bsxfun(@times,Ay_pol_pos,P),1);
%             F_pos=bsxfun(@times,S_dist_hat_pos',nansum(bsxfun(@times,P,log2(bsxfun(@rdivide,sprime_on_s,sum(bsxfun(@times,S_dist_hat_pos',sprime_on_s),2)))),3)); %#ok<AGROW>
%             Ay_pol_pos=Ay_pol_pos+bsxfun(@times,Ay_pol_pos/(t+1),bsxfun(@minus,F_pos,sum(Ay_pol_pos.*F_pos,1))); %#ok<AGROW>
%             PI_T=squeeze(sprime_on_s)';
%             PI_sprime=PI_T*S_dist_hat;
%             PI_joint=bsxfun(@times,S_dist_hat_pos',PI_T);
%             converged= mean(abs(Ay_pol_pos(:)-Ay_pol_old(:)))<0.0000001;
%           end
%         end        
      end
      U_history=zeros(T,max(U_index));
      avgU_history=U_history;
      PI_calculated_history=zeros(T,max(PI_index));
      
      % test nav power
      if nav_cutoff
        opt_nav_dist=zeros(1,N);
        nav_dist=zeros(ceil(T/10),N,K);
        Q_nav_dist=zeros(ceil(T/10),N,K);
        
        navpolicy=test_navigation(P,nav_cutoff);
        for target=1:N
          abp=P;
          abp(:,target,:)=0;
          abp(:,target,target)=1;
          navpol=permute(navpolicy(:,target,:),[3,1,2]);
          transp=squeeze(sum(bsxfun(@times,abp,navpol),1));
          state_dist=ones(1,N)/(N-1);
          state_dist(target)=0;
          for t_count=1:nav_cutoff
            old_dist=state_dist;
            state_dist=state_dist*transp;
            opt_nav_dist(1,target)=opt_nav_dist(1,target)+t_count*(state_dist(target)-old_dist(target));
          end
          opt_nav_dist(1,target)=opt_nav_dist(1,target)+(1-state_dist(target))*nav_cutoff;
        end
      end
      %end test nav
      
      %test reward
      if rew_cutoff
        opt_rew_gains=zeros(1,reward_trials);
        rew_gains=zeros(T,reward_trials,K);
        rewardstruct=rand(1,1,N,1,reward_trials)*2-1;
        for trial=1:reward_trials
          reward=rewardstruct(:,:,:,:,trial);
          % rewpolicy [logic cond A, cond S, 1, strat]
          rewV=zeros(1,1,N); % [1, 1, cond S, strat]
          rewVtrue=rewV; % [1, 1, out S, strat]
          for navt=1:rew_cutoff
            rewQ=sum(bsxfun(@times,reward+rewV,P),3); % [cond A*, cond S*, 1, strat]
            rewV=max(rewQ,[],1); % [1, cond S, 1, strat]
            rewpolicy=double(bsxfun(@eq,rewQ,rewV)); % [logic cond A*, cond S, 1, strat]
            rewpolicy=bsxfun(@rdivide,rewpolicy,sum(rewpolicy,1)); % [cond A, cond S, 1, strat]
            rewV=permute(rewV,[1,3,2,4]); % [1, 1, cond S, strat]
            rewQtrue=sum(bsxfun(@times,reward+rewVtrue,P),3); % [cond A*, cond S*, 1, strat]
            rewVtrue=permute(sum(rewQtrue.*rewpolicy,1),[1,3,2,4,5]); % [1, 1, cond S, strat]
          end
          opt_rew_gains(trial)=(zerosN1'+1/N)*squeeze(rewVtrue);
        end
        rewardstruct=repmat(rewardstruct,[1,1,1,K,1]);
      end
      %end test reward
      
      % EXPLORE
      for t=1:T-1
        for i=1:K
          k=strategies(i);
          
          % choose A according to strategy
          % 1-UPIG, 2-GPIG, 3-VIPIG, 4-VI+PIG,5-QVIPIG,
          % 6-USur,7-GSur,8-VISur,9-VI+Sure, 10-Q_sur,
          % 11 - SFI PI (optimize by gradient approach)
          % 12 - SFI PI (optimize by gradient approach given true model)
          % 13 - Combined approach          
          
          switch k  
            case 30 % random action baseline control
              A(t,i)=pregen_random_a(t);
              
            case {1,6} %Unembodied Strategies
              utility=U_store(:,:,U_index(i));
              [r,s]=find(utility==max(utility(:)));
              U_history(t,U_index(i))=max(utility(:));
              avgU_history(t,U_index(i))=mean(utility(:));
              ind=length(r);
              if ind ~= 1
                ind=randsample(ind,1);
              end
              A(t,i)=r(ind);
              S(t,i)=s(ind);
              
            case {2,7} % Greedy Strategies
              max_utility=max(U_store(:,S(t,i),U_index(i)));
              r_ind=find(U_store(:,S(t,i),U_index(i))==max_utility);
              U_history(t,U_index(i))=max_utility;
              avgU_history(t,U_index(i))=mean(U_store(:,S(t,i),U_index(i)));
              if length(r_ind)~=1
                r_ind=randsample(r_ind,1);
              end
              A(t,i)=r_ind;
              
            case {3,4,8,9,13} % Value-Iterated Strategies
              if t==1
                A(t,i)=randsample(M,1);
              else
                %value iteration
                if any(k==[4,9])
                  CMC=P;
                else
                  CMC=Q(:,:,:,i);
                end
                utility=U_store(:,:,U_index(i));
                VI_value=zeros11N;
                
                
                  for stratcount=1:10
                    valuecross=utility+discount*sum(bsxfun(@times,CMC,VI_value),3); %VI_store(:,s)+discount*permute(VI_P(:,s,:),[1,3,2])*(old_VI_value);
                    VI_value(:)=max(valuecross);
                  end
                if mod(t,50)==0
                  Value_history(:,:,t/50,VI_index(i))=valuecross;
                end
                if k==13 && VI_value(S(t,i))<PI_gain*PI_store(PI_index(i))
                  A(t,i)=randsample(M,1,true,Ay_pol(:,S(t,i),Ay_index(i)));
                  PI_chosen_history(t,PIvsPIG_index(i))=1; %#ok<AGROW>
                else
                  U_history(t,U_index(i))=VI_value(S(t,i));
                  avgU_history(t,U_index(i))=mean(valuecross(:,S(t,i)));
                  r_ind=find(valuecross(:,S(t,i))==VI_value(S(t,i)));
                  if length(r_ind)==1
                    A(t,i)=r_ind;
                  else
                    A(t,i)=randsample(r_ind,1);
                  end
                end
              end
                            
            case {5,10} % Q-Learning Strategies
              max_utility=max(Q_store(:,S(t,i),Q_index(i)));
              r_ind=find(Q_store(:,S(t,i),Q_index(i))==max_utility);
              U_history(t,U_index(i))=max_utility;
              avgU_history(t,U_index(i))=mean(Q_store(:,S(t,i),Q_index(i)));
              if length(r_ind)~=1
                r_ind=randsample(r_ind,1);
              end
              A(t,i)=r_ind;
              
            case {11,12}
              A(t,i)=randsample(M,1,true,Ay_pol(:,S(t,i),Ay_index(i)));
%             case 12
%               A(t,i)=randsample(M,1,true,Ay_pol_pos(:,S(t,i)));
              
          end
          
          % Determine Resultant State
          S(t+1,i)=sum(pregen_rand(t)>P_cumulation(A(t,i),S(t,i),:))+1;
          
          % Determine histograms and distributions of conditionals
          S_hist(S(t,i),i)=S_hist(S(t,i),i)+1;                             % add the state just acted in to histogram [for stratey i]
          J_hist(A(t,i),S(t,i),i)=J_hist(A(t,i),S(t,i),i)+1;               % add state and action just performed in to joint histogram
          W_hist(A(t,i),S(t,i),S(t+1,i),i)=W_hist(A(t,i),S(t,i),S(t+1,i),i)+1;
          S_dist(:,i)=S_hist(:,i)/t;                                       % normalize state histogram to get state distribution
          J_dist(:,:,i)=J_hist(:,:,i)/t;                                   % normalize joint histogram to get joint distribution
          
          if t>1
            Spair_prior((S(t,i)-1)*N+S(t-1,i),i)=Spair_prior((S(t,i)-1)*N+S(t-1,i),i)+1;
            Spair_post((S(t+1,i)-1)*N+S(t,i),i)=Spair_post((S(t+1,i)-1)*N+S(t,i),i)+1;
            Spair_joint((S(t,i)-1)*N+S(t-1,i),(S(t+1,i)-1)*N+S(t,i),i)=Spair_joint((S(t,i)-1)*N+S(t-1,i),(S(t+1,i)-1)*N+S(t,i),i)+1;
            temp=Spair_joint(:,:,i);
            PI2_history(t+1,i)=-nansum(Spair_prior(:,i)/(t-1).*log2(Spair_prior(:,i)/(t-1)))-nansum(Spair_post(:,i)/(t-1).*log2(Spair_post(:,i)/(t-1)))+nansum(temp(:)/(t-1).*log2(temp(:)/(t-1)));

          end
          
          
          S_hist_new(:,i)=S_hist(:,i);
          S_hist_new(S(t+1,i),i)=S_hist_new(S(t+1,i),i)+1;
          
          S_joint(S(t,i),S(t+1,i),i)=S_joint(S(t,i),S(t+1,i),i)+1;
          
          
          % Calculate additional histograms for empirical PI
          if any(k==[17,18,19])
            S_2step_hist(S(t,i),S(t+1,i),emp2_index(i))=S_2step_hist(S(t,i),S(t+1,i),emp2_index(i))+1; %#ok<AGROW>
            if t>1
              S_2step_joint(S(t-1,i),S(t,i),S(t,i),S(t+1,i),emp2_index(i))=S_2step_joint(S(t-1,i),S(t,i),S(t,i),S(t+1,i),emp2_index(i))+1; %#ok<AGROW>
            end
          end
          
          % update Q
          if any(k==[6,7,8,9,10])
            Q_old=Q(A(t,i),S(t,i),:,i);
          end
            J_smoothed(A(t,i),S(t,i),i)=J_smoothed(A(t,i),S(t,i),i)+1;
            W_smoothed(A(t,i),S(t,i),S(t+1,i),i)=W_smoothed(A(t,i),S(t,i),S(t+1,i),i)+1;
            Q(A(t,i),S(t,i),:,i)=W_smoothed(A(t,i),S(t,i),:,i)./J_smoothed(A(t,i),S(t,i),i);
            if ~isempty( Q(A(t,i),S(t,i),~OutSize(A(t,i),S(t,i),:),i))
              'asdf'
            end
          % end update Q
          
          % Calculate PI
          Sprime_hist=S_hist_new(:,i);
          Sprime_hist(S(1,i))=Sprime_hist(S(1,i))-1;
          S_joint_copy=S_joint(:,:,i);
          PI_history(t+1,i)=-nansum(S_hist(:,i)/t.*log2(S_hist(:,i)/t))-nansum(Sprime_hist/t.*log2(Sprime_hist/t))+nansum(S_joint_copy(:)/t.*log2(S_joint_copy(:)/t));          
          
          % Update Utility Stores
          switch k
            case {1,2,3,4,5,13} % Information Gain
                projected_W_smoothed=bsxfun(@plus,squeeze(W_smoothed(A(t,i),S(t,i),:,i)),eyeN); % [N(future) x N(hypothesized outcome)]
                projected_W_smoothed(:,~Mask(A(t,i),S(t,i),:))=0;
                projected_Q=projected_W_smoothed/(J_smoothed(A(t,i),S(t,i),i)+1);
                U_store(A(t,i),S(t,i),U_index(i))=nansum(projected_Q.*log2(bsxfun(@rdivide,projected_Q,squeeze(Q(A(t,i),S(t,i),:,i)))),1)*squeeze(Q(A(t,i),S(t,i),:,i)); %#ok<AGROW> %sum_s+ q(s+|D)*(sum_s p(s|D,s+)log2 (p(s|D,s+)/p(s|D)))
                              
            case {6,7,8,9,10} % Surprise
              U_store(A(t,i),S(t,i),U_index(i))=nansum(Q(A(t,i),S(t,i),:,i).*log2(Q(A(t,i),S(t,i),:,i)./Q_old)); %#ok<AGROW>
          end
          
          % Update Coordination Related Variables
          if any(k==[5,10]) %Q-learning
            Q_store(A(t,i),S(t,i),Q_index(i))=(1-Qbeta)*Q_store(A(t,i),S(t,i),Q_index(i))+Qbeta*(U_store(A(t,i),S(t,i),U_index(i))+Qgamma*max(Q_store(:,S(t+1),Q_index(i)))); %#ok<AGROW>
            if mod(t,50)==0
              Q_history(:,:,t/50,Q_index(i))=Q_store(:,:,Q_index(i));
            end
          end
          
          % Update PI policy
          if any(k==[11,13])
            ind=Ay_index(i);
            S_dist_hat(:,ind)=(t*S_dist_hat(:,ind))/(t+1); %#ok<AGROW>
            S_dist_hat(S(t+1,i),ind)=S_dist_hat(S(t+1,i),ind)+1/(t+1); %#ok<AGROW>
            switch k
                case 12
                    CMC=P;
                otherwise
                    CMC=Q(:,:,:,i);
            end
            sprime_on_s=sum(bsxfun(@times,Ay_pol(:,:,ind),CMC),1);
            F(:,:,ind)=bsxfun(@times,S_dist_hat(:,ind)',nansum(bsxfun(@times,CMC,log2(bsxfun(@rdivide,sprime_on_s,sum(bsxfun(@times,S_dist_hat(:,ind)',sprime_on_s),2)))),3)); %#ok<AGROW>
            Ay_pol(:,:,ind)=Ay_pol(:,:,ind)+bsxfun(@times,Ay_pol(:,:,ind)/(t+1),bsxfun(@minus,F(:,:,ind),sum(Ay_pol(:,:,ind).*F(:,:,ind),1))); %#ok<AGROW>
            PI_T=squeeze(sprime_on_s)';
            PI_sprime=PI_T*S_dist_hat(:,ind);
            PI_joint=bsxfun(@times,S_dist_hat(:,ind)',PI_T);
            PI_store(PI_index(i))=-nansum(S_dist_hat(:,ind).*log2(S_dist_hat(:,ind)))-nansum(PI_sprime(:).*log2(PI_sprime(:)))+nansum(PI_joint(:).*log2(PI_joint(:))); %#ok<AGROW>
            PI_calculated_history(t,PI_index(i))=PI_store(PI_index(i));
          end
          
          for target=1:N
            Q_nav(A(t,i),S(t,i),target,i)=(1-Qnavbeta)*Q_nav(A(t,i),S(t,i),target,i)+Qnavbeta*(double(S(t+1,i)==target)+Qnavgamma*max(Q_nav(:,S(t+1,i),target,i))); %#ok<AGROW>
          end 
        end % strategies loop
        
        % test nav power
        if nav_cutoff && mod(t,10)==0
          navpolicy=test_navigation(Q,nav_cutoff);
          for target=1:N
            abp=P;
            abp(:,target,:)=0;
            abp(:,target,target)=1;
            for strat=1:K
              navpol=permute(navpolicy(:,target,:,strat),[3,1,2,4]);
              transp=squeeze(sum(bsxfun(@times,abp,navpol),1));
              state_dist=ones(1,N)/(N-1);
              state_dist(target)=0;
              for t_count=1:nav_cutoff
                old_dist=state_dist;
                state_dist=state_dist*transp;
                nav_dist(t/10,target,strat)=nav_dist(t/10,target,strat)+t_count*(state_dist(target)-old_dist(target));
              end
              nav_dist(t/10,target,strat)=nav_dist(t/10,target,strat)+(1-state_dist(target))*nav_cutoff;
            end
          end
          
          navpolicy=double(bsxfun(@eq,Q_nav,max(Q_nav,[],1)));
          navpolicy=bsxfun(@rdivide,navpolicy,sum(navpolicy,1));
          for target=1:N
            abp=P;
            abp(:,target,:)=0;
            abp(:,target,target)=1;
            for strat=1:K
              navpol=navpolicy(:,:,target,strat);
              transp=squeeze(sum(bsxfun(@times,abp,navpol),1));
              state_dist=ones(1,N)/(N-1);
              state_dist(target)=0;
              for t_count=1:nav_cutoff
                old_dist=state_dist;
                state_dist=state_dist*transp;
                Q_nav_dist(t/10,target,strat)=Q_nav_dist(t/10,target,strat)+t_count*(state_dist(target)-old_dist(target));
              end
              Q_nav_dist(t/10,target,strat)=Q_nav_dist(t/10,target,strat)+(1-state_dist(target))*nav_cutoff;
            end
          end
          
        end
        %end test nav
        
        %test reward
        for trial=1:reward_trials
          reward=rewardstruct(:,:,:,:,trial);
          rewV=zeros(1,1,N,K); % [1, 1, cond S, strat]
          rewVtrue=rewV; % [1, 1, out S, strat]
          for navt=1:rew_cutoff
            rewQ=sum(bsxfun(@times,reward+rewV,Q),3); % [cond A*, cond S*, 1, strat]
            rewV=max(rewQ,[],1); % [1, cond S, 1, strat]
            rewpolicy=double(bsxfun(@eq,rewQ,rewV)); % [logic cond A*, cond S, 1, strat]
            rewpolicy=bsxfun(@rdivide,rewpolicy,sum(rewpolicy,1)); % [cond A, cond S, 1, strat]
            rewV=permute(rewV,[1,3,2,4]); % [1, 1, cond S, strat]
            rewQtrue=sum(bsxfun(@times,reward+rewVtrue,Krep_P),3); % [cond A*, cond S*, 1, strat]
            rewVtrue=permute(sum(rewQtrue.*rewpolicy,1),[1,3,2,4,5]); % [1, 1, cond S, strat]
          end
          rew_gains(t,trial,:)=(zerosN1'+1/N)*squeeze(rewVtrue);
        end
        %end test reward
        
        % calculate missing information
        KL=Krep_P.*log2(Krep_P./Q).*KMask;                 % KL = KL contribution of each parameter to KL-divergence between P and Q [M x N x N x K]
        KL(isnan(KL))=0;
        KL_history(t+1,:)=sum(sum(sum(KL,3),2),1);
        
        
      end % time loop
      
      % save results to obj
      obj.strategies=strategies;
      obj.A_history=A;
      obj.S_history=S;
      obj.KL_history=KL_history;
      obj.Q=Q;
      obj.Mask=Mask;
      obj.T=T;
      obj.lambda=lambda;
      obj.context=context;
      obj.discount=discount;
      obj.nav_cutoff=nav_cutoff;
      obj.epsilon=epsilon;
      obj.rew_cutoff=rew_cutoff;
      obj.infer_lambda=infer_lambda;
      obj.U_index=U_index;
      obj.U_history=U_history;
      obj.avgU_history=avgU_history;
      
      obj.U_store=U_store;
      
      obj.PI_history=PI_history;
      obj.PI2_history=PI2_history;
      obj.PI_calculated_history=PI_calculated_history;
      if any(strategies==3)
        obj.Value_history=Value_history;
        obj.VI_index=VI_index;
      end
      if any(strategies==13)
        
        obj.PI_chosen_history=PI_chosen_history;
      end
      if any(strategies==5)||any(strategies==10)
        
        obj.Q_store=Q_store;
        obj.Q_history=Q_history;
      end

      
      obj.PIvsPIG_index=PIvsPIG_index;
      obj.PI_index=PI_index;
      obj.PI_gain=PI_gain;
%       if any(strategies==12)
%         obj.Ay_pol_pos=Ay_pol_pos;
%       end
      if any(strategies==11)||any(strategies==13) || any(strategies==12)
      end
      if nav_cutoff
        obj.nav_dist=nav_dist;
        obj.Q_nav_dist=Q_nav_dist;
        obj.opt_nav_dist=opt_nav_dist;
      end
      if rew_cutoff
        obj.rew_gains=rew_gains;
        obj.opt_rew_gains=opt_rew_gains;
      end
      if infer_lambda
        obj.contextual_inference_history=squeeze(contextual_inference_history);
      end
    end %explore function
    
    
    
  end
  
end



function [nav_pol] = test_navigation(P_hat,runlength)
[M,N,holder,K]=size(P_hat); %#ok<ASGLU>
navP_hat=repmat(P_hat,[1,1,1,1,N]); % [cond A, cond S, out S, strat, target S]
policy=ones(M,N,1,K,N)/M; % [logic cond A, cond S, 1, strat, targ S]
reward=-ones(M,N,N,K,N); % [cond A, cond S, out S, strat, targ S]
V=zeros(1,1,N,K,N); % [1, 1, out S, strat, targS]
for s=1:N
  navP_hat(:,s,:,:,s)=0;
  navP_hat(:,s,s,:,s)=1;
  reward(:,s,s,:,s)=0;
end
Prewsum=sum(reward.*navP_hat,3);

for count=1:runlength
  Q=Prewsum+sum(bsxfun(@times,V,navP_hat),3);% [cond A*, cond S*, 1, strat, targ S]
  Vtemp=max(Q,[],1); % [1, cond S, 1, strat, targ S]
  policy=double(bsxfun(@eq,Q,Vtemp)); % [logic cond A*, cond S, 1, strat, targ S]
  policy=bsxfun(@rdivide,policy,sum(policy,1)); % [cond A, cond S, 1, strat, targ S]
  V=permute(Vtemp,[1,3,2,4,5]); % [1, 1, cond S, strat, targ S]
end

nav_pol=squeeze(permute(policy,[3,2,5,1,4])); % [current S, targ S, action choice, strat]

end
