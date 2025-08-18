function timeMsg = executionTime (time)

% Calculate 'hours', 'minutes' and 'seconds'
hours   = floor (time/3600);
minutes = floor (mod(time,3600)/60);
seconds = round (mod(time,60));

% Check whether 'hours' is greater than '0'
if hours > 0  % if so
    % Generate 'timeMsg' as follows
    timeMsg = sprintf ('%d hours, %d minutes, %d seconds', hours, minutes, seconds);
else  % if 'hours == 0'
    % Check whether 'minutes' is greater than '0'
    if minutes > 0  % if so
        % Generate 'timeMsg' as follows
        timeMsg = sprintf ('%d minutes, %d seconds', minutes, seconds);
    else  % if 'minutes == 0'
        % Recalculate 'seconds' as follows
        seconds = floor(mod(time,60));
        % Calculate 'msec'
        msec = round (1000 * (mod(time,60) - seconds));
        % Check whether 'seconds' is greater than '0'
        if seconds > 0  % if so
            % Generate 'timeMsg' as follows
            timeMsg = sprintf ('%d seconds, %d msec', seconds, msec);
        else  % if 'seconds == 0'
            timeMsg = sprintf ('%d msec', msec);
        end
    end
end