function [indClosest, radSearch, dist] = find_neighbours (X, Xneig, radInit)

% Initialise variables
indIntoBB = [];
radSearch = 0;

while isempty (indIntoBB)
    % Update search radio
    radSearch = radSearch + radInit;
    
    %%% X axis
    % Define search range for X coordinate
    xRange = [X(1)-radSearch, X(1)+radSearch];
    % Take points whose value for X coordinate is within the range specified by 'xRange'
    indX = (Xneig(:,1) >= xRange(1)) & (Xneig(:,1) <= xRange(2));
    % Take the indices of points in 'Xreig' satisfying 'xRange'
    indIntoBB = find (indX);
    pointsIntoBB = Xneig (indIntoBB, :);
    
    % Check whether some points satisfy previous conditions
    if ~isempty (indIntoBB)
        %%% Y axis
        % Define search range for Y coordinate
        yRange = [X(2)-radSearch, X(2)+radSearch];
        % Take points whose value for Y coordinate is within the range specified by 'yRange'
        indY = (pointsIntoBB(:,2) >= yRange(1)) & (pointsIntoBB(:,2) <= yRange(2));
        % Take the indices of points in 'Xreig' satisfying 'xRange' and 'yRange'
        indIntoBB = indIntoBB (indY);
        pointsIntoBB = Xneig (indIntoBB, :);
        
        % Check whether some points satisfy previous conditions
        if ~isempty (indIntoBB)
            %%% Z axis
            % Define search range for Z coordinate
            zRange = [X(3)-radSearch, X(3)+radSearch];
            % Take points whose value for Z coordinate is within the range specified by 'zRange'
            indZ = (pointsIntoBB(:,3) >= zRange(1)) & (pointsIntoBB(:,3) <= zRange(2));
            % Take the indices of points in 'Xreig' satisfying 'xRange', 'yRange' and 'zRange'
            indIntoBB = indIntoBB (indZ);
            pointsIntoBB = Xneig (indIntoBB, :);
            
            %%% Distance
            % Check whether some points satisfy previous conditions
            if ~isempty (indIntoBB)
                % Calculate distance to the closest node
                [dist, ind] = pdist2 (pointsIntoBB, X, 'euclidean', 'Smallest', 1);
                % Check whether the distance to the closest node into bounding box is smaller or equal to 'radSearch'
                if dist <= radSearch  % if so
                    % Take the index of the closest node
                    indClosest = indIntoBB (ind);
                else  % if it is greater
                    % Set 'indIntoBB' as an empty matrix to keep searching
                    indIntoBB = [];
                end
            end
        end
    end
end

return