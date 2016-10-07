library("proxy", lib="/udd/aroyer/local/Rlib")
library("reshape", lib="/udd/aroyer/local/Rlib")
library("dtw", lib="/udd/aroyer/local/Rlib")
library("ggplot2", lib="/udd/aroyer/local/Rlib")
#library("dtw")
#library("ggplot2")
library(reshape)
options(show.error.messages=FALSE) 

###Compute a DWT with constraints
compute_dtw <- function(query, ref, unconstrained=0){
    if (unconstrained == 0) {        
        alignment <- dtw(query, ref, keep=TRUE, window.type="slantedband", window.size=5, step.pattern = rabinerJuangStepPattern(2, "c", TRUE));
        return(alignment)
    } else if (unconstrained == 1) { 
        alignment <- dtw(query, ref, keep=TRUE, window.type="slantedband", window.size=5);
        return(alignment)
    }
    else {
        alignment <- dtw(query, ref, keep=TRUE);
        return(alignment)
    }
}


###Plot an alignment with local cost
plot_dtw <- function(namquery, namref, alignment, output){
    lcm <- alignment$localCostMatrix
    nref <- ncol(lcm)
    nquery <- nrow(lcm)
    lcm <- melt(matrix(lcm, ncol=ncol(lcm)))
    df = data.frame(query = alignment$index1, ref = alignment$index2)
    heatmap <- ggplot(lcm, aes(y=X1,x=X2)) + geom_tile(aes(fill = value), colour = "white", position = "dodge") + scale_fill_gradient(low = "white", high = "chartreuse3") + geom_line(data=df, aes(x=ref, y = query), color="red3") + geom_line(data=data.frame(x1=c(1, nref), x2=c(1, nquery)), aes(x=x1, y = x2), color="blue", alpha=0.5, linetype="dotted") + scale_x_continuous(limits = c(0, nref+1), expand=c(0,0)) + scale_y_continuous(limits = c(0, nquery+1), expand=c(0,0))  + theme_classic()+ labs(x=sprintf("%s (%d)", namref, nref), y=sprintf("%s (%d)", namquery, nquery)) + ggtitle(sprintf("Ref %s x Query %s", namref, namquery))                                                                
    ggsave(output)
    dev.off()
}

plot_and_compute_dtw <- function(query, ref, namquery, namref, output) {
    alignment <- compute_dtw(query, ref)
    plot_dtw(namquery, namref, alignment, output)
}		  

                                        #directory <- "/home/amelie/Repositories/stage_inria_2015/Code/Data/Audio/20samples(50)/Uncompressed_Features/MFCC_0_D_A_Z_39/";
                                        #files <- list.files(path = directory, pattern = "*.txt", full.names=TRUE);

###First plot some things
                                        #for (x in files) {
                                        #        featx <- read.table(x)
                                        #        for (y in files) {
                                        #            y = files[2]        
                                        #            featy <- read.table(y)
                                        #            alignment <- compute_dtw(featx, featy)
                                        #            plot_dtw(basename(x) , basename(y) , alignment, 'truc.pdf')
                                        #            stopifnot(0 > 1)
                                        #        }
                                        #    }
