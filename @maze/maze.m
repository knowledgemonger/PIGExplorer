

classdef maze < handle
   properties
      size % number of rooms along one side of the square maze
      p = struct('full', NaN,'collapsed', NaN) % probabilities describing paths through the maze.
      % full = 5 dimensional: response x position x position [position = h x v]
      % collapsed = 3 dimension: response x room x room
      walls = struct('horizontal',NaN,'vertical',NaN);% array of structures representing training sessions
      ports = struct('up',NaN,'down',NaN,'left',NaN, 'right', NaN);
      context = struct('full', NaN, 'collapsed', NaN);
   end
   
   methods (Access = 'public')
      
      
      function obj=maze(size)
         obj.size=size;
         obj.p.full(1:4,1:size,1:size,1:size,1:size)=0;
         obj.p.collapsed(1:4,1:size*size,1:size*size)=0;
         obj.walls.horizontal(1:size,1:size+1)=true;
         obj.walls.verticle(1:size+1,1:size)=true;
         obj.ports.up(1:size,1:size)=false;
         obj.ports.down(1:size,1:size)=false;
         obj.ports.left(1:size,1:size)=false;
         obj.ports.right(1:size,1:size)=false;
         obj.context.full=ones(4,size,size);
         obj.context.collapsed=ones(4,size*size);
      end
      
      function generate(obj,number_ports,absorb)
         if nargin<3
            absorb=1;
         end
         size=obj.size;
         sizesize=size*size;
         ax=randsample(size,1);
         ay=randsample(size,1);
         walls.horizontal(1:size,1:size+1)=true;
         walls.vertical(1:size+1,1:size)=true;
         path(1:size,1:size)=false;
         path(1,1)=true;
         border(1:size,1:size)=false;
         border(1,1)=true;
         free_sides(1:size,1:size)=4;
         free_sides(1,1:size)=free_sides(1,1:size)-1;
         free_sides(size,1:size)=free_sides(size,1:size)-1;
         free_sides(1:size,1)=free_sides(1:size,1)-1;
         free_sides(1:size,size)=free_sides(1:size,size)-1;
         free_sides(1,2)=free_sides(1,2)-1;
         free_sides(2,1)=free_sides(2,1)-1;
         
         context=obj.context;
         
         for i=2:sizesize
            [x,y]=find(border);
            n=randsample(length(x),1);
            removable(1:4)=true; %up right down left
            if y(n)==1 %up
               removable(1)=false;
            elseif path(x(n),y(n)-1)
               removable(1)=false;
            end
            if x(n)==size %right
               removable(2)=false;
            elseif path(x(n)+1,y(n))
               removable(2)=false;
            end
            if y(n)==size %down
               removable(3)=false;
            elseif path(x(n),y(n)+1)
               removable(3)=false;
            end
            if x(n)==1 %left
               removable(4)=false;
            elseif path(x(n)-1,y(n))
               removable(4)=false;
            end
            if sum(removable)==1
               dir=find(removable);
            else
               dir=randsample(find(removable),1);
            end
            if dir==1 %up
               walls.horizontal(x(n),y(n))=false;
               path(x(n),y(n)-1)=true;
               if free_sides(x(n),y(n)-1)~=0
                  border(x(n),y(n)-1)=true;
               end
               free_sides(x(n),y(n))=free_sides(x(n),y(n))-1;
               if free_sides(x(n),y(n))==0
                  border(x(n),y(n))=false;
               end
               if y(n)-1~=1
                  free_sides(x(n),y(n)-2)=free_sides(x(n),y(n)-2)-1;
                  if border(x(n),y(n)-2)&& free_sides(x(n),y(n)-2)==0
                     border(x(n),y(n)-2)=false;
                  end
               end
               if x(n)~=size
                  free_sides(x(n)+1,y(n)-1)=free_sides(x(n)+1,y(n)-1)-1;
                  if border(x(n)+1,y(n)-1)&& free_sides(x(n)+1,y(n)-1)==0
                     border(x(n)+1,y(n)-1)=false;
                  end
               end
               if x(n)~=1
                  free_sides(x(n)-1,y(n)-1)=free_sides(x(n)-1,y(n)-1)-1;
                  if border(x(n)-1,y(n)-1) && free_sides(x(n)-1,y(n)-1)==0
                     border(x(n)-1,y(n)-1)=false;
                  end
               end
            end
            
            if dir==2 %right
               walls.vertical(x(n)+1,y(n))=false;
               path(x(n)+1,y(n))=true;
               if free_sides(x(n)+1,y(n))~=0
                  border(x(n)+1,y(n))=true;
               end
               free_sides(x(n),y(n))=free_sides(x(n),y(n))-1;
               if free_sides(x(n),y(n))==0
                  border(x(n),y(n))=false;
               end
               if y(n)~=1
                  free_sides(x(n)+1,y(n)-1)=free_sides(x(n)+1,y(n)-1)-1;
                  if border(x(n)+1,y(n)-1)&& free_sides(x(n)+1,y(n)-1)==0
                     border(x(n)+1,y(n)-1)=false;
                  end
               end
               if x(n)+1~=size
                  free_sides(x(n)+2,y(n))=free_sides(x(n)+2,y(n))-1;
                  if border(x(n)+2,y(n))&& free_sides(x(n)+2,y(n))==0
                     border(x(n)+2,y(n))=false;
                  end
               end
               if y(n)~=size
                  free_sides(x(n)+1,y(n)+1)=free_sides(x(n)+1,y(n)+1)-1;
                  if border(x(n)+1,y(n)+1) && free_sides(x(n)+1,y(n)+1)==0
                     border(x(n)+1,y(n)+1)=false;
                  end
               end
            end
            
            if dir==3 %down
               walls.horizontal(x(n),y(n)+1)=false;
               path(x(n),y(n)+1)=true;
               if free_sides(x(n),y(n)+1)~=0
                  border(x(n),y(n)+1)=true;
               end
               free_sides(x(n),y(n))=free_sides(x(n),y(n))-1;
               if free_sides(x(n),y(n))==0
                  border(x(n),y(n))=false;
               end
               if y(n)+1~=size
                  free_sides(x(n),y(n)+2)=free_sides(x(n),y(n)+2)-1;
                  if border(x(n),y(n)+2)&& free_sides(x(n),y(n)+2)==0
                     border(x(n),y(n)+2)=false;
                  end
               end
               if x(n)~=size
                  free_sides(x(n)+1,y(n)+1)=free_sides(x(n)+1,y(n)+1)-1;
                  if border(x(n)+1,y(n)+1)&& free_sides(x(n)+1,y(n)+1)==0
                     border(x(n)+1,y(n)+1)=false;
                  end
               end
               if x(n)~=1
                  free_sides(x(n)-1,y(n)+1)=free_sides(x(n)-1,y(n)+1)-1;
                  if border(x(n)-1,y(n)+1) && free_sides(x(n)-1,y(n)+1)==0
                     border(x(n)-1,y(n)+1)=false;
                  end
               end
            end
            
            if dir==4 %left
               walls.vertical(x(n),y(n))=false;
               path(x(n)-1,y(n))=true;
               if free_sides(x(n)-1,y(n))~=0
                  border(x(n)-1,y(n))=true;
               end
               free_sides(x(n),y(n))=free_sides(x(n),y(n))-1;
               if free_sides(x(n),y(n))==0
                  border(x(n),y(n))=false;
               end
               if y(n)~=1
                  free_sides(x(n)-1,y(n)-1)=free_sides(x(n)-1,y(n)-1)-1;
                  if border(x(n)-1,y(n)-1)&& free_sides(x(n)-1,y(n)-1)==0
                     border(x(n)-1,y(n)-1)=false;
                  end
               end
               if x(n)-1~=1
                  free_sides(x(n)-2,y(n))=free_sides(x(n)-2,y(n))-1;
                  if border(x(n)-2,y(n))&& free_sides(x(n)-2,y(n))==0
                     border(x(n)-2,y(n))=false;
                  end
               end
               if y(n)~=size
                  free_sides(x(n)-1,y(n)+1)=free_sides(x(n)-1,y(n)+1)-1;
                  if border(x(n)-1,y(n)+1) && free_sides(x(n)-1,y(n)+1)==0
                     border(x(n)-1,y(n)+1)=false;
                  end
               end
            end
         end
         
         ports.up(1:size,1:size)=false;
         ports.down(1:size,1:size)=false;
         ports.left(1:size,1:size)=false;
         ports.right(1:size,1:size)=false;
         [ux,uy]=find(walls.horizontal(1:size,1:size));
         [rx,ry]=find(walls.vertical(2:size+1,1:size));
         [dx,dy]=find(walls.horizontal(1:size,2:size+1));
         [lx,ly]=find(walls.vertical(1:size,1:size));
         nu=length(ux);
         nr=length(rx);
         nd=length(dx);
         nl=length(lx);
         faces=nu+nr+nd+nl;
         port_index=randsample(faces,number_ports);
         for i=1:number_ports
            if port_index(i)<= nu
               ports.up(ux(port_index(i)),uy(port_index(i)))=true;
            elseif port_index(i)<= nu+nr
               ports.right(rx(port_index(i)-nu),ry(port_index(i)-nu))=true;
            elseif port_index(i)<= nu+nr+nd
               ports.down(dx(port_index(i)-nu-nr),dy(port_index(i)-nu-nr))=true;
            else
               ports.left(lx(port_index(i)-nu-nr-nd),ly(port_index(i)-nu-nr-nd))=true;
               
            end
         end
         
         p.full(1:4,1:size,1:size,1:size,1:size)=0;
         context.full=ones(4,size,size);
         for x=1:size
            for y=1:size
               % if up
               if ports.up(x,y)
                  if absorb==1
                     p.full(1,x,y,ax,ay)=1;
                  elseif absorb==0
                     p.full(1,x,y,:,:)=1/sizesize;
                     context.full(1,x,y)=2;
                  elseif rand<absorb
                     p.full(1,x,y,ax,ay)=1;
                  else
                     p.full(1,x,y,:,:)=1/sizesize;
                     context.full(1,x,y)=2;
                  end
               elseif walls.horizontal(x,y)
                  p.full(1,x,y,x,y)=1;
               else
                  p.full(1,x,y,x,y-1)=1;
               end
               
               % if right
               if ports.right(x,y)
                  if absorb==1
                     p.full(2,x,y,ax,ay)=1;
                  elseif absorb==0
                     p.full(2,x,y,:,:)=1/sizesize;
                     context.full(2,x,y)=2;
                  elseif rand<absorb
                     p.full(2,x,y,ax,ay)=1;
                  else
                     p.full(2,x,y,:,:)=1/sizesize;
                     context.full(2,x,y)=2;
                  end
               elseif walls.vertical(x+1,y)
                  p.full(2,x,y,x,y)=1;
               else
                  p.full(2,x,y,x+1,y)=1;
               end
               
               % if down
               if ports.down(x,y)
                  if absorb==1
                     p.full(3,x,y,ax,ay)=1;
                  elseif absorb==0
                     p.full(3,x,y,:,:)=1/sizesize;
                     context.full(3,x,y)=2;
                  elseif rand<absorb
                     p.full(3,x,y,ax,ay)=1;
                  else
                     p.full(3,x,y,:,:)=1/sizesize;
                     context.full(3,x,y)=2;
                  end
               elseif walls.horizontal(x,y+1)
                  p.full(3,x,y,x,y)=1;
               else
                  p.full(3,x,y,x,y+1)=1;
               end
               
               % if left
               if ports.left(x,y)
                  if absorb==1
                     p.full(4,x,y,ax,ay)=1;
                  elseif absorb==0
                     p.full(4,x,y,:,:)=1/sizesize;
                     context.full(4,x,y)=2;
                  elseif rand<absorb
                     p.full(4,x,y,ax,ay)=1;
                  else
                     p.full(4,x,y,:,:)=1/sizesize;
                     context.full(4,x,y)=2;
                  end
               elseif walls.vertical(x,y)
                  p.full(4,x,y,x,y)=1;
               else
                  p.full(4,x,y,x-1,y)=1;
               end
               
            end
         end
         
         p.collapsed(1:4,1:sizesize,1:sizesize)=NaN;
         for x1=1:size
            for y1=1:size
               for x2=1:size
                  for y2=1:size
                     p.collapsed(:,x1+size*(y1-1),x2+size*(y2-1))=p.full(:,x1,y1,x2,y2);
                     context.collapsed(:,x1+size*(y1-1))=context.full(:,x1,y1);
                  end
               end
            end
         end
         
         obj.walls=walls;
         obj.ports=ports;
         obj.p=p;
         obj.context=context;
         
      end
      
      function illustrate(obj,figure_id,robotx,roboty,robotdir,name,field)
         %figure(figure_id);
         size=obj.size;
         walls=obj.walls;
         ports=obj.ports;
         context=obj.context;
         if nargin<7
            field=ones(size);
            field(1,1)=0;
         end
         colormap Gray;
         imagesc(field');
         [x,y]=find(walls.horizontal);
         line([x'-0.55;x'+0.55],[y'-0.5;y'-0.5],'linewidth',3,'color','black');
         [x,y]=find(walls.vertical);
         line([x'-0.5;x'-0.5],[y'-0.55;y'+0.55],'linewidth',3,'color','black');
         [x,y]=find(ports.up & squeeze(context.full(1,:,:)==1));
         line([x'-0.3;x'+0.3],[y'-0.4;y'-0.4],'linewidth',1.5,'color','blue');
         [x,y]=find(ports.up & squeeze(context.full(1,:,:)==2));
         line([x'-0.3;x'+0.3],[y'-0.4;y'-0.4],'linewidth',1.5,'color','red');
         [x,y]=find(ports.down & squeeze(context.full(3,:,:)==1));
         line([x'-0.3;x'+0.3],[y'+0.4;y'+0.4],'linewidth',1.5,'color','blue');
         [x,y]=find(ports.down & squeeze(context.full(3,:,:)==2));
         line([x'-0.3;x'+0.3],[y'+0.4;y'+0.4],'linewidth',1.5,'color','red');
         [x,y]=find(ports.left & squeeze(context.full(4,:,:)==1));
         line([x'-0.4;x'-0.4],[y'-0.3;y'+0.3],'linewidth',1.5,'color','blue');
         [x,y]=find(ports.left & squeeze(context.full(4,:,:)==2));
         line([x'-0.4;x'-0.4],[y'-0.3;y'+0.3],'linewidth',1.5,'color','red');
         [x,y]=find(ports.right & squeeze(context.full(2,:,:)==1));
         line([x'+0.4;x'+0.4],[y'-0.3;y'+0.3],'linewidth',1.5,'color','blue');
         [x,y]=find(ports.right & squeeze(context.full(2,:,:)==2));
         line([x'+0.4;x'+0.4],[y'-0.3;y'+0.3],'linewidth',1.5,'color','red');
         %             for x=1:size
         %                 for y=1:size
         %                     text(x-0.05,y-0.03,int2str(x+size*(y-1)),'color',[0.7,0.7,0.7]);
         %                 end
         %             end
         if nargin>3
            if nargin<5
               name='BOB';
            end
            text(robotx,roboty,name,'HorizontalAlignment','center','VerticalAlignment','middle','Color',[0,0.7,0],'Rotation',robotdir,'fontsize',20);
         end
      end
      
      
      function illustrate_action(obj,figure_id,robotx,roboty,robotdir,name,field)
         %figure(figure_id);
         size=obj.size;
         walls=obj.walls;
         ports=obj.ports;
         context=obj.context;
         if nargin<7
            field=ones(size);
            field(1,1)=0;
         end
         colormap Gray;
%          imagesc(field');
         [absx,absy]=find(squeeze(squeeze(sum(sum(sum(obj.p.full,3),2),1)))==max(max(squeeze(squeeze(sum(sum(sum(obj.p.full,3),2),1))))),1);
         rectangle('Position',[absx-0.75,6.25-absy,0.5,0.5],'Curvature',[1,1],'FaceColor','none','edgecolor','c','linewidth',1.5)
         rectangle('Position',[absx-0.65,6.35-absy,0.3,0.3],'Curvature',[1,1],'FaceColor','none','edgecolor','c','linewidth',1.5)
         rectangle('Position',[absx-0.85,6.15-absy,0.7,0.7],'Curvature',[1,1],'FaceColor','none','edgecolor','c','linewidth',1.5)
         [x,y]=find(walls.horizontal);
         x=x-0.5;
         y=7-y+0.5;
         line([x'-0.5;x'+0.5],[y'-0.5;y'-0.5],'linewidth',2,'color','black');
         [x,y]=find(walls.vertical);
         x=x-0.5;
         y=6-y+0.5;
         line([x'-0.5;x'-0.5],[y'-0.5;y'+0.5],'linewidth',2,'color','black');
         [x,y]=find(ports.up & squeeze(context.full(1,:,:)==1));
         x=x-0.5;
         y=6-y+0.5;
         line([x'-0.3;x'+0.3],[y'+0.4;y'+0.4],'linewidth',1.5,'color','blue');
         [x,y]=find(ports.up & squeeze(context.full(1,:,:)==2));
         x=x-0.5;
         y=6-y+0.5;
         line([x'-0.3;x'+0.3],[y'+0.4;y'+0.4],'linewidth',1.5,'color','red');
         [x,y]=find(ports.down & squeeze(context.full(3,:,:)==1));
         x=x-0.5;
         y=6-y+0.5;
         line([x'-0.3;x'+0.3],[y'-0.4;y'-0.4],'linewidth',1.5,'color','blue');
         [x,y]=find(ports.down & squeeze(context.full(3,:,:)==2));
         x=x-0.5;
         y=6-y+0.5;
         line([x'-0.3;x'+0.3],[y'-0.4;y'-0.4],'linewidth',1.5,'color','red');
         [x,y]=find(ports.left & squeeze(context.full(4,:,:)==1));
         x=x-0.5;
         y=6-y+0.5;
         line([x'-0.4;x'-0.4],[y'-0.3;y'+0.3],'linewidth',1.5,'color','blue');
         [x,y]=find(ports.left & squeeze(context.full(4,:,:)==2));
         x=x-0.5;
         y=6-y+0.5;
         line([x'-0.4;x'-0.4],[y'-0.3;y'+0.3],'linewidth',1.5,'color','red');
         [x,y]=find(ports.right & squeeze(context.full(2,:,:)==1));
         x=x-0.5;
         y=6-y+0.5;
         line([x'+0.4;x'+0.4],[y'-0.3;y'+0.3],'linewidth',1.5,'color','blue');
         [x,y]=find(ports.right & squeeze(context.full(2,:,:)==2));
         x=x-0.5;
         y=6-y+0.5;
         line([x'+0.4;x'+0.4],[y'-0.3;y'+0.3],'linewidth',1.5,'color','red');
         %             for x=1:size
         %                 for y=1:size
         %                     text(x-0.05,y-0.03,int2str(x+size*(y-1)),'color',[0.7,0.7,0.7]);
         %                 end
         %             end
         if nargin>3
            if nargin<6
               name='\downarrow';
            end
            for x=1:length(robotx)
            text(robotx(x),roboty(x),name,'HorizontalAlignment','center','VerticalAlignment','middle','Color',[0,0.7,0],'Rotation',robotdir(x),'fontsize',20);
            end
         end
      end
   end
      
      
      
   end
