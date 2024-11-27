<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
        <xsl:output indent="no" method="text"/>

        <xsl:template match="/">
                <xsl:for-each select="//cellExpression/subAssay[@type='human' and @subtype='human cell lines']/data/assayImage//imageUrl">
                        <xsl:value-of select="text()"/>
                        <xsl:text> </xsl:text>
                        <xsl:value-of select="../../..//cellLine/text()"/>
                        <xsl:text>&#xa;</xsl:text><!-- new line -->
                </xsl:for-each>
        </xsl:template>

</xsl:stylesheet>