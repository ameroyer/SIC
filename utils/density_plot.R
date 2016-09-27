args<-commandArgs(TRUE)
library(ggplot2)

#---------------------------------Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


#---------------------Arguments
file = args[1]
gt = args[2]
pdff = args[3]
title = args[4]
pdf(file=pdff)

#----------------------Data
data1 = read.table(file)
data1$id <- 'sim'
data2 = read.table(gt)
data2$id <- 'gt'
alldata <- rbind(data1,data2)

#--------------------Similarities histogram
# Histogram
p1 <- ggplot(data1, aes(x=V1)) + geom_histogram(colour='#51CE9A', fill='#6CE5AE', alpha = 0.6) + ggtitle(title) + xlab("Unnormalized similarities") + ylab("Counts")

# With density plot
p2 <- ggplot(data1, aes(x=V1)) + geom_histogram(colour='#51CE9A', fill='#6CE5AE', alpha = 0.6, aes(y=..density..)) + geom_density(alpha = .5, colour = '#6B6994', fill="#C8BEE5") + ylab("Density")  + theme(axis.text.x=element_blank(),axis.title.x=element_blank())

#multiplot(p1, p2)

#----------------------Similarity with ground-truth
# Histogram
p3 <- ggplot(alldata, aes(V1, fill=id)) + geom_histogram(position="identity", alpha = .6) + xlab("Unnormalized similarities") + theme(axis.text.y=element_blank(), axis.title.y=element_blank()) + ggtitle('')

# Density plot
p4 <- ggplot(alldata, aes(x=V1, fill=id)) + geom_density(aes(y=..count..), alpha = 0.3)+   geom_vline(data = data1, aes(xintercept=mean(V1, na.rm=T)), color="red", linetype="dashed") +  geom_vline(data = data1, aes(xintercept=as.numeric(names(which.max(table(V1))))), color="blue", linetype="dashed") + theme(axis.text.x=element_blank(), axis.text.y=element_blank(), axis.title.x=element_blank(), axis.title.y=element_blank())

multiplot(p1, p2, p3, p4, cols = 2)
dev.off() 
