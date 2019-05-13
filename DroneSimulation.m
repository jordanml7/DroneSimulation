% Jordan Lueck
% COMP150 Probabilistic Robotics
% Homework 2 - Image-Based Particle Filter for Drone Localization
% Simulation Environment

map = imread('BayMap.png');
unit = 50;
m = 50;
N = 500;
% Set W between 0 and 1
% Currently set to ignore GPS measurements (given results of testing)
W = 0;
theta = linspace(0,2*pi);
xrange = length(map(1,:,:));
yrange = length(map(:,1,:));

x = round([random('unif',m,xrange-m); random('unif',m,yrange-m)]);
[p, p_edges] = generate_particles(N,xrange,yrange,m);

figure('Position',[0 0 xrange yrange])
while(true)
    imagesc(map);
    hold on
    
    reference = measurement_with_noise(x,map,m);
    gps_x = gps_coords(x);
    mov_vec = movement_vector(x,unit,m,xrange,yrange);

    [p, avgx, avgy, p_edges] = run_particle_filter(p,map,N,m,reference,...
        gps_x,mov_vec,unit,xrange,yrange,p_edges,W);
    x = agent_movement_step(x,mov_vec,unit);
    plot_agent(x,m,theta);

    text(100,50,strcat({'Actual Position: '},num2str(x(1)),{', '},num2str(x(2))));
    text(100,80,strcat({'Average Particle Position: '},num2str(avgx),{', '},num2str(avgy)));
    hold off

    status = input('','s');
    if status == 'q'
        break
    end
end

net_diff = sqrt((x(1)-avgx)^2+(x(2)-avgy)^2);
close all

%% Simulation Functions

function reference = measurement_with_noise(x,map,m)
    ref_mu = 0;
    ref_sig2 = 10;
    
    row_noise = round(normrnd(ref_mu,ref_sig2));
    col_noise = round(normrnd(ref_mu,ref_sig2));
    COLS = (x(1) - m/2 + col_noise) : (x(1) + m/2 + col_noise - 1);
    ROWS = (x(2) - m/2 + row_noise) : (x(2) + m/2 + row_noise - 1);
    
    % code to visualize the reference location (with added noise)
    %{
    theta = linspace(0,2*pi);
    ref_xc = COLS(m/2) + (m/2)*cos(theta);
    ref_yc = ROWS(m/2) + (m/2)*sin(theta);
    plot(ref_xc,ref_yc,'DisplayName','Reference Location',...
        'LineWidth',2,'Color','yellow');
    %}
    
    reference = map(ROWS,COLS,:);
    % code that adds noise to pixel colors if desired
    %{
    col_mu = 0;
    col_sig2 = 5;
    for i = 1:m
        for j = 1:m
            reference(i,j,1) = reference(i,j,1) + round(normrnd(col_mu,col_sig2));
            reference(i,j,2) = reference(i,j,2) + round(normrnd(col_mu,col_sig2));
            reference(i,j,3) = reference(i,j,3) + round(normrnd(col_mu,col_sig2));
        end
    end
    %}
end

function mov_vec = movement_vector(x,unit,m,xrange,yrange)
    theta = round(random('unif',0,2*pi));
    xrand = round(unit*cos(theta));
    yrand = round(unit*sin(theta));
   
    while (x(1)+xrand > xrange-m || x(1)+xrand < m+1 ...
        || x(2) + yrand > yrange-m || x(2) + yrand < m+1)
        theta = round(random('unif',0,2*pi));
        xrand = round(unit*cos(theta));
        yrand = round(unit*sin(theta));
    end
    
    mov_vec = [xrand; yrand];
end

function x = agent_movement_step(x,mov_vec,unit)
    mov_mu = 0;
    mov_sig2 = round(unit/10);

    x(1) = x(1) + mov_vec(1) + round(normrnd(mov_mu,mov_sig2));
    x(2) = x(2) + mov_vec(2) + round(normrnd(mov_mu,mov_sig2));
end

function plot_agent(x,m,theta)
    for i = 2:0.5:5
        xc = x(1) + (m/i)*cos(theta);
        yc = x(2) + (m/i)*sin(theta);
        plot(xc,yc,'LineWidth',3);
    end
end

function gps_x = gps_coords(x)
    gps_x = x;
    gps_mu = 0;
    gps_sig2 = 20;
    
    gps_x(1) = x(1) + round(normrnd(gps_mu,gps_sig2));
    gps_x(2) = x(2) + round(normrnd(gps_mu,gps_sig2));
    
    % code to visualize gps location (with added noise)
    %{
    theta = linspace(0,2*pi);
    gps_xc = gps_x(1) + 2*cos(theta);
    gps_yc = gps_x(2) + 2*sin(theta);
    plot(gps_xc,gps_yc,'LineWidth',4);
    %}
end

%% Particle Filter Functions

function [p, p_edges] = generate_particles(N,xrange,yrange,m)
    p = zeros(2,N);
    p_edges = zeros(1,N);
    for i = 1:N
        p(:,i) = round([random('unif',m,xrange-m); random('unif',m,yrange-m)]);
    end
end

function [p,avgx,avgy,p_edges] = run_particle_filter(p, map, N, m, ...
    reference, gps_x, mov_vec, unit, xrange, yrange, p_edges, W)
    meas_weights = zeros(1,N);
    gps_weights = zeros(1,N);
    theta = linspace(0,2*pi);
    for i = 1:N
        z = measurement_without_noise(p(:,i),map,m);
        if p_edges(i)
            meas_weights(i) = 0;
            gps_weights(i) = 0;
        else
            meas_weights(i) = particle_approximator_hist(z,reference);
            gps_weights(i) = gps_weight_approximator(p(:,i),gps_x);
        end
    end
    gps_weights = gps_weights/norm(gps_weights);
    gps_weights = 1 - gps_weights;
    meas_weights = meas_weights/norm(meas_weights);
    weights = W*gps_weights + (1-W)*meas_weights;
    
    p = resample(p,weights,N);
    p_edges = zeros(1,N);
    for i = 1:N
        [p(:,i), p_edges(i)] = particle_movement_step(p(:,i),mov_vec,unit,xrange,yrange,m,p_edges(i));
    end
    [avgx, avgy] = draw_particles(p,N,m,theta);
end

function [p, p_edge] = particle_movement_step(x,mov_vec,unit,xrange,yrange,m,p_edge)
    mov_mu = 0;
    mov_sig2 = round(unit/10);

    p(1) = x(1) + mov_vec(1) + round(normrnd(mov_mu,mov_sig2));
    p(2) = x(2) + mov_vec(2) + round(normrnd(mov_mu,mov_sig2));
    
    % catch particles that are moved outside map with movement vector & set
    % weights to zero
    if p(1) < m+1 || p(1) > xrange-m || p(2) < m+1 || p(2) > yrange-m
        p(1) = x(1);
        p(2) = x(2);
        p_edge = 1;
    end
end

function reference = measurement_without_noise(x,map,m)
    COLS = (x(1) - m/2) : (x(1) + m/2 - 1);
    ROWS = (x(2) - m/2) : (x(2) + m/2 - 1);
    
    reference = map(ROWS,COLS,:);
end

function weight = particle_approximator_avgdiff(z,x)
    if size(x) ~= size(z)
        fprintf('Measurement Window Invalid, exiting...\n');
        return;
    end
    x = double(x);
    z = double(z);
    
    diff = mean(mean(mean(imabsdiff(x,z))));
    
    weight = 1 - diff/255;
end

function weight = particle_approximator_hist(z,x)
    % compute measurement color hist values
    [measR, ~] = imhist(z(:,:,1));
    [measG, ~] = imhist(z(:,:,2));
    [measB, ~] = imhist(z(:,:,3));
    % compute reference color hist values
    [refR, ~] = imhist(x(:,:,1));
    [refG, ~] = imhist(x(:,:,2));
    [refB, ~] = imhist(x(:,:,3));
    
    R_corr = abs(corr2(measR,refR));
    G_corr = abs(corr2(measG,refG));
    B_corr = abs(corr2(measB,refB));
    
    weight = mean([R_corr G_corr B_corr]);
end

function weight = particle_approximator_graymap(z,x)
    x = rgb2gray(x);
    z = rgb2gray(z);
    % compute measurement hist values
    [meas, ~] = imhist(z);
    % compute reference hist values
    [ref, ~] = imhist(x);
    
    weight = abs(corr2(meas,ref));
end

function gps_weight = gps_weight_approximator(p,gps)
    gps_weight = sqrt((p(1)-gps(1))^2+(p(2)-gps(2))^2);
end

function p = resample(p0,weights,N)
    indices = randsample(N,N,true,weights);
    p = zeros(2,N);
    for i = 1:N
        p(:,i) = p0(:,indices(i));
    end
end

function [avgx, avgy] = draw_particles(p,N,m,theta)
    avgx = 0;
    avgy = 0;
    for i = 1:N
        pxc = p(1,i) + (m/6)*cos(theta);
        avgx = avgx + p(1,i);
        pyc = p(2,i) + (m/6)*sin(theta);
        avgy = avgy + p(2,i);
        plot(pxc,pyc,'LineWidth',2,'Color','white');
    end
    avgx = avgx / N;
    avgy = avgy / N;
end