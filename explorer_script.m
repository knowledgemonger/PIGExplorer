numworlds=25;
numsteps=2001;
strategies=[1,30,3,11:13];
roomswidth=6;
size_s=roomswidth^2;
size_r=4;
numstrats=length(strategies);
test_nav=0;
test_freq=[25:25:100,150:50:500,600:100:1000,1500:500:5000];
context=1;
discount=0.95;
blue_ports=30;
red_ports=0;
ports=blue_ports+red_ports;
options.numDiff=1;
options.Display='off';
dir_maze=zeros(size_r,size_s,size_s);
funh=@(x,varargin) EKL_average_mazetrnasition(x,varargin);
lambda=0.25;

set=1;
PIgain=0.1;
navcutoff=25;
rewcutoff=25;


for w=1:numworlds
  w
  tic
  maze_worlds(w)=maze(roomswidth);
  maze_worlds(w).generate(ports,1);
  drunk_maze=bsxfun(@plus,maze_worlds(w).p.collapsed*(1-0.8/3),mean(maze_worlds(w).p.collapsed,1)*0.8/3);
  Mask=drunk_maze~=0;
  dir_maze(:)=0;
  for a=1:4
     for s=1:36
        while sum(dir_maze(a,s,:))~=1
         dir_maze(a,s,:)=randg(lambda,[1,1,36]);
         dir_maze(a,s,~Mask(a,s,:))=0;
         dir_maze=bsxfun(@rdivide,dir_maze,sum(dir_maze,3));
        end
     end
  end
  for a=1:4
     for s=1:36
        mix=randperm(36);
        imix(mix)=1:36;
        [temp,ix]=sort(drunk_maze(a,s,mix));
        iix(ix)=1:36;
        [dsort,temp]=sort(dir_maze(a,s,:));
        dir_maze(a,s,:)=dsort(iix(imix));
     end
  end
  maze_explorers(w)=explorer(dir_maze);
  maze_explorers(w).explore(strategies,numsteps,lambda,Mask,context,discount, PIgain, 0,0,0);
  toc
end
save( ['SavedExplorer_',int2str(set)], '-v7.3'); 