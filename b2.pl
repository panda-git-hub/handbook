#!/usr/bin/perl
$tgtYM = $ARGV[0];
chomp $tgtYM;
$total_profit = 0;
@csvs = glob "./data/$tgtYM/*.csv";
for $csv ( @csvs ) {
        &init_parms();
        open ( IN, $csv );
        while ( $l = <IN> ) {
                chomp $l;
                if ( $l =~ /^202/ ) {
                        @p = split( /,/, $l );
                        if ( $p[0] =~ /22:00:00$/ ) {
                                &bin_start();
                        }
                        elsif ( $status eq "CLEAR" ) {
                                &bin_start();
                        }
                 
                        &bin_rate()     if ( $status ne "NONE" );
                        $status = "END" if ( $p[0] =~ /23:59:00$/ );
                        &bin_output()   if ( $status ne "NONE" );
                 
                        if ( $status eq "END" ) {
                                &bin_end();
                                last;
                        }
                }
        }
        close ( IN );
        print "\n\n" . "-" x 30 . "\n\n";
        print "RESULT : $sum_profit\n\n";
        $total_profit += $sum_profit;
}

print "TOTALRESULT: $total_profit\n\n";


sub init_parms() {
        $status = "NONE";
        $sum_profit = 0;
        $lv = 0;
        $ch = 0;
}


sub bin_start() {
        $status = "ENTRY";
        $open   = sprintf( "%.3f", $p[1] );
        $high   = sprintf( "%.3f", $p[6] );
        $low    = sprintf( "%.3f", $p[7] );
        $target = sprintf( "%.3f", $open - 0.1 );
}


sub bin_end() {
        $status = "END";
        $rslt   = $open - $close;
        $sum_profit += sprintf( "%.1f", $rslt * 100 );
}


sub bin_rate() {
        $high  = sprintf( "%.3f", $p[6] ) if ( $p[6] > $high );
        $low   = sprintf( "%.3f", $p[7] ) if ( $p[7] < $low  );
        $close = $p[8];
#       $gain  = sprintf( "%.3f", $open - $low  );
        $loss  = sprintf( "%.3f", $high - $open );
        $status = "LOSS" if( $loss >= 0.1 );

        if ( $loss >= 1 ) {
                $target = sprintf( "%.3f", $open + 0.443 );
                if ( $lv < 6 ) {
                        $lv = 6;
                        $ch = 1;
                }
        }
        elsif ( $loss >= 0.8 ) {
                $target = sprintf( "%.3f", $open + 0.35 );
                if ( $lv < 5 ) {
                        $lv = 5;
                        $ch = 1;
                }
        }
        elsif ( $loss >= 0.6 ) {
                $target = sprintf( "%.3f", $open + 0.26 );
                if ( $lv < 4 ) {
                        $lv = 4;
                        $ch = 1;
                }
        }
        elsif ( $loss >= 0.4 ) {
                $target = sprintf( "%.3f", $open + 0.175 );
                if ( $lv < 3 ) {
                        $lv = 3;
                        $ch = 1;
                }
        }
        elsif ( $loss >= 0.2 ) {
                $target = sprintf( "%.3f", $open + 0.1 );
                if ( $lv < 2 ) {
                        $lv = 2;
                        $ch = 1;
                }
        }
        elsif ( $loss >= 0.1 ) {
                $target = sprintf( "%.3f", $open + 0.05 );
                if ( $lv == 0 ) {
                        $lv = 1;
                        $ch = 1;
                }
        }
         
        if ( $p[7] <= $target and $ch == 0 ) {
                $sum_profit += 10 if ( $status eq "ENTRY" );
                $status = "CLEAR";
                $lv = 0;
        }
 
        $ch = 0;
}



sub bin_output() {
        print "$l\n";
        print "$p[0]\t$status\t$open\t$high\t$low\t$close  /  $loss / $target\n\n\n";
}




__END__
